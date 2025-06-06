# examples/run_grpo_genrm.py
import argparse
import json
import os
import pprint
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.genrm_environment import GenRMEnvironment
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def helpsteer3_genrm_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process HelpSteer3 data for GenRM training."""
    
    # Extract the prompt (which contains the full formatted prompt with responses)
    prompt = datum_dict.get("prompt", "")
    
    # Extract metadata from the args field
    args = datum_dict.get("args", {})
    num_responses = args.get("num_responses", 1)
    helpfulness_1 = args.get("helpfulness_1", None)
    helpfulness_2 = args.get("helpfulness_2", None)
    preference_ranking = args.get("preference_ranking", None)
    
    # Extract system prompt if present
    system_prompt = datum_dict.get("system_prompt", "")
    
    # Create message log
    message_log = []
    user_message = {
        "role": "user",
        "content": prompt,
    }
    message: list[str] = tokenizer.apply_chat_template(  # type: ignore
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message[0]
    message_log.append(user_message)
    

    # Calculate total length
    total_length = sum(len(msg["token_ids"]) for msg in message_log)
    
    # Check if we need to truncate
    loss_multiplier = 1.0
    if total_length > max_seq_length:
        # Truncate messages
        for msg in message_log:
            msg["token_ids"] = msg["token_ids"][: min(4, max_seq_length // len(message_log)) ]
        loss_multiplier = 0.0
    
    # Prepare metadata for environment
    metadata = {
        "num_responses": num_responses,
        "helpfulness_1": helpfulness_1,
        "helpfulness_2": helpfulness_2,
        "preference_ranking": preference_ranking,
    }

    return DatumSpec(
        message_log=message_log,
        length=total_length,
        extra_env_info=metadata,
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name="genrm",
    )


class HelpSteer3LocalDataset(Dataset):
    """Dataset for loading HelpSteer3 data from local JSONL files."""
    
    def __init__(self, data_path: str, split: str = "train"):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.split = split
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Add task_name for compatibility with AllTaskProcessedDataset
        item = self.data[idx].copy()
        item["task_name"] = "genrm"
        return item


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training for GenRM")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--train-data", 
        type=str, 
        default="/home/yizhujiao/dataset/hs3_genrm/train_data.jsonl",
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="/home/yizhujiao/dataset/hs3_genrm/val_data.jsonl",
        help="Path to validation data JSONL file"
    )

    args, overrides = parser.parse_known_args()
    return args, overrides


def setup_data(
    tokenizer: PreTrainedTokenizerBase,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    train_data_path: str,
    val_data_path: str,
) -> tuple[AllTaskProcessedDataset, Optional[AllTaskProcessedDataset], dict, dict]:
    """Set up data for GenRM training."""
    
    print("\n▶ Setting up data...")
    
    # Create task spec for GenRM
    genrm_task_spec = TaskDataSpec(
        task_name="genrm",
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
    )
    
    # Load local datasets
    train_dataset = HelpSteer3LocalDataset(train_data_path, split="train")
    val_dataset = HelpSteer3LocalDataset(val_data_path, split="validation") if val_data_path else None
    
    # ISSUE 1 FIX: Use defaultdict to handle the task processor mapping correctly
    # This ensures that any task_name gets mapped to the genrm processor
    task_data_processors = defaultdict(lambda: (genrm_task_spec, helpsteer3_genrm_data_processor))
    task_data_processors["genrm"] = (genrm_task_spec, helpsteer3_genrm_data_processor)
    
    # Initialize GenRM environment
    genrm_env = GenRMEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.genrm_environment.GenRMEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs.get("genrm", {}))
    
    # Create processed datasets
    processed_train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        genrm_task_spec,  # default_task_data_spec
        task_data_processors,  # task_data_processors dict
        max_seq_length=data_config["max_input_seq_length"],
    )
    
    processed_val_dataset = None
    if val_dataset:
        processed_val_dataset = AllTaskProcessedDataset(
            val_dataset,
            tokenizer,
            genrm_task_spec,  # default_task_data_spec
            task_data_processors,  # task_data_processors dict
            max_seq_length=data_config["max_input_seq_length"],
        )
    
    task_to_env = defaultdict(lambda: genrm_env)
    task_to_env["genrm"] = genrm_env
    
    # For validation, use the same environment mapping
    val_task_to_env = defaultdict(lambda: genrm_env)
    val_task_to_env["genrm"] = genrm_env
    
    return processed_train_dataset, processed_val_dataset, task_to_env, val_task_to_env



def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_genrm_1B.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    init_ray()

    # Setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # Setup data
    dataset, val_dataset, task_to_env, val_task_to_env = setup_data(
        tokenizer, 
        config["data"], 
        config["env"],
        args.train_data,
        args.val_data,
    )

    # Setup GRPO training
    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # Run GRPO training
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()