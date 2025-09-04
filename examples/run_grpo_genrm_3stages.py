# Three-Stage GenRM Training Script - Modified to load Stage 1 results
import argparse
import os
import json
import torch
import pprint
import numpy as np
from tqdm import tqdm
from typing import Any, Optional, Dict, List
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf
from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.hf_datasets.reward_benchmarks import HelpSteer3LocalDataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

# Import our three-stage components
from nemo_rl.environments.genrm_environment_3stages import (
    ThreeStageGenRMEnvironment,
    ThreeStageMetadata,
    format_factcheck_prompt,
    format_vanilla_genrm_prompt
)

# ========================= STAGE 1 RESULTS LOADER =========================

class Stage1ResultsDataset(torch.utils.data.Dataset):
    """Dataset that loads stage 1 results from JSON file."""
    
    def __init__(self, stage1_results_path: str, task_name: str = "three_stage_genrm"):
        self.data = []
        self.task_name = task_name
        
        print(f"Loading stage 1 results from: {stage1_results_path}")
        with open(stage1_results_path, 'r') as f:
            stage1_results = json.load(f)
        
        # Convert stage 1 results to dataset format
        for result in stage1_results:
            if "predicted_scores" in result and "predicted_ranking" in result:
                metadata = result["metadata"]
                
                # Create dataset entry with stage 1 results embedded
                entry = {
                    "context": metadata["context"],
                    "response1": metadata["response1"], 
                    "response2": metadata["response2"],
                    "num_responses": metadata.get("num_responses", 2),
                    "helpfulness_1": metadata.get("helpfulness_1"),
                    "helpfulness_2": metadata.get("helpfulness_2"),
                    "preference_ranking": metadata.get("preference_ranking"),
                    
                    # Stage 1 results
                    "vanilla_scores": result["predicted_scores"],
                    "vanilla_ranking": result["predicted_ranking"],
                    "vanilla_response": result["prediction"],
                    
                    "task_name": task_name
                }

                self.data.append(entry)
        
        print(f"Loaded {len(self.data)} valid stage 1 results")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]




# ========================= DATA PROCESSOR FOR STAGE 2+ =========================

def three_stage_genrm_data_processor_from_stage1(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process data that may have stage 1 results, supporting both training and validation modes."""
    
    # Extract data
    context = datum_dict.get("context", "")
    response1 = datum_dict.get("response1", "")
    response2 = datum_dict.get("response2", "")
    
    num_responses = datum_dict.get("num_responses", 2)
    helpfulness_1 = datum_dict.get("helpfulness_1", None)
    helpfulness_2 = datum_dict.get("helpfulness_2", None)
    preference_ranking = datum_dict.get("preference_ranking", None)
    
    # Stage 1 results (may be pre-computed or None for validation)
    vanilla_scores = datum_dict.get("vanilla_scores", [])
    vanilla_ranking = datum_dict.get("vanilla_ranking", None)
    vanilla_response = datum_dict.get("vanilla_response", "")
    
    # Determine starting stage based on whether we have vanilla scores
    if vanilla_scores and vanilla_ranking is not None:
        # Pre-computed vanilla scores - start from fact-checking (stage 2)
        starting_stage = "factcheck"
        prompt = format_factcheck_prompt(context, response1, response2)
    else:
        # No vanilla scores - start from vanilla GenRM (stage 1) 
        starting_stage = "vanilla"
        prompt = format_vanilla_genrm_prompt(context, response1, response2)
    
    # Create message log
    message_log = []
    user_message = {
        "role": "user",
        "content": prompt,
    }
    
    # Apply chat template
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message  # Update with formatted content
    message_log.append(user_message)
    
    # Calculate total length
    total_length = sum(len(msg["token_ids"]) for msg in message_log)
    
    # Check if we need to truncate
    loss_multiplier = 1.0
    if total_length > max_seq_length:
        for msg in message_log:
            msg["token_ids"] = msg["token_ids"][:min(4, max_seq_length // len(message_log))]
        loss_multiplier = 0.0
    
    # Prepare metadata for three-stage environment
    metadata: ThreeStageMetadata = {
        "num_responses": num_responses,
        "helpfulness_1": helpfulness_1,
        "helpfulness_2": helpfulness_2,
        "preference_ranking": preference_ranking,
        "stage": starting_stage,  # Start from appropriate stage
        "vanilla_scores": vanilla_scores if vanilla_scores else None,
        "vanilla_ranking": vanilla_ranking,
        "vanilla_response": vanilla_response,
        "factcheck_results": None,
        "context": context,
        "response1": response1,
        "response2": response2,
    }
    
    return DatumSpec(
        message_log=message_log,
        length=total_length,
        extra_env_info=metadata,
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name="three_stage_genrm",
    )

# ========================= DATA SETUP =========================

def setup_three_stage_data_from_stage1(tokenizer, data_config, env_configs, stage1_results_path):
    """Setup data for three-stage GenRM training starting from stage 1 results."""
    
    print(f"\n▶ Setting up three-stage data from stage 1 results...")
    
    # Create task spec for three-stage GenRM
    three_stage_task_spec = TaskDataSpec(
        task_name="three_stage_genrm",
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
    )
    
    # Load stage 1 results dataset for training
    train_dataset = Stage1ResultsDataset(
        stage1_results_path, 
        task_name="three_stage_genrm"
    )
    
    # For validation, use HelpSteer3 validation set going through full 3-stage process
    val_dataset = None
    val_data_path = data_config.get("val_data_path")
    if val_data_path and os.path.exists(val_data_path):
        print(f"Loading validation data from HelpSteer3: {val_data_path}")
        val_dataset = HelpSteer3LocalDataset(
            val_data_path,
            task_name="three_stage_genrm",
            split="validation"
        )
    else:
        print("No validation data path provided or file not found")
    
    # Setup task data processors
    task_data_processors = defaultdict(lambda: (three_stage_task_spec, three_stage_genrm_data_processor_from_stage1))
    task_data_processors["three_stage_genrm"] = (three_stage_task_spec, three_stage_genrm_data_processor_from_stage1)
    
    # Setup three-stage environment
    from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env

    three_stage_env = ThreeStageGenRMEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.genrm_environment_3stages.ThreeStageGenRMEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs.get("three_stage_genrm", {}))
    
    # Update task to environment mapping
    task_to_env = defaultdict(lambda: three_stage_env)
    task_to_env["three_stage_genrm"] = three_stage_env
    
    val_task_to_env = defaultdict(lambda: three_stage_env)
    val_task_to_env["three_stage_genrm"] = three_stage_env

    # Create processed datasets
    processed_train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        three_stage_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    
    processed_val_dataset = None
    if val_dataset:
        processed_val_dataset = AllTaskProcessedDataset(
            val_dataset,
            tokenizer,
            three_stage_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
        print(f"Created validation dataset with {len(processed_val_dataset)} samples")
    
    return processed_train_dataset, processed_val_dataset, task_to_env, val_task_to_env

# ========================= TRAINING SETUP =========================

def setup_three_stage_training(config, tokenizer, dataset, val_dataset):
    """Setup three-stage GenRM training pipeline."""
    
    # Setup base components using the standard GRPO setup
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
    
    return (
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
    )

# ========================= MAIN TRAINING SCRIPT =========================

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training for Three-Stage GenRM from Stage 1 results")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--stage1-results", type=str, default="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/NeMo-RL/results/1grpo_hs3_16K_step240_clip_max_0.28_qwen3_14b_lr_2e-6_temp_1_kl_0.001_grpo_bs_256_rollout_64_num_prompts_128_r0_base/outputs/step_45_hs3train_results.json", 
        help="Path to JSON file containing stage 1 results"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides

def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_genrm_3stages.yaml"
        )

    # Validate stage 1 results file
    if not os.path.exists(args.stage1_results):
        raise FileNotFoundError(f"Stage 1 results file not found: {args.stage1_results}")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    print(f"Loading stage 1 results from: {args.stage1_results}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)
    
    # Get the next experiment directory
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    
    # Initialize Ray
    from nemo_rl.distributed.virtual_cluster import init_ray
    init_ray()
    
    # Setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # Setup data from stage 1 results
    train_dataset, val_dataset, task_to_env, val_task_to_env = setup_three_stage_data_from_stage1(
        tokenizer, 
        config["data"], 
        config["env"],
        args.stage1_results
    )
    
    # Debug: Print first example from the dataset
    if len(train_dataset) > 0:
        print("\n[DEBUG] First example from dataset:")
        first_example = train_dataset[0]
        print(f"  Keys: {list(first_example.keys())}")
        for key, value in first_example.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"  {key}: {value[:200]}...")
            elif key == "extra_env_info" and isinstance(value, dict):
                print(f"  {key} keys: {list(value.keys())}")
                print(f"    stage: {value.get('stage')}")
                print(f"    vanilla_scores: {value.get('vanilla_scores')}")
                print(f"    vanilla_ranking: {value.get('vanilla_ranking')}")
            else:
                print(f"  {key}: {value}")

    # Setup three-stage training
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
    ) = setup_three_stage_training(config, tokenizer, train_dataset, val_dataset)
    
    # Run GRPO training with task environments
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,  # Pass task_to_env instead of a single env
        val_task_to_env,  # Pass val_task_to_env
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )

if __name__ == "__main__":
    main()