# Two-Stage GenRM Integration Guide
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
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

# Import our two-stage components
from nemo_rl.environments.genrm_environment_w_fact import (
    TwoStageFactCheckEnvironment, 
    TwoStageMetadata,
    format_factcheck_stage_prompt,
    format_scoring_stage_prompt,
    parse_scoring_response
)

# ========================= DATA PROCESSOR =========================

def two_stage_genrm_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process HelpSteer3 data for two-stage GenRM training."""
    
    # Extract the original prompt content and parse it
    original_prompt = datum_dict.get("prompt", "")
    
    # Extract the original prompt content and parse it
    original_prompt = datum_dict.get("prompt", "")
    
    # FIXED: Extract metadata from the args field (the actual data structure)
    args = datum_dict.get("args", {})
    num_responses = args.get("num_responses", 2)
    helpfulness_1 = args.get("helpfulness_1", None)
    helpfulness_2 = args.get("helpfulness_2", None)
    preference_ranking = args.get("preference_ranking", None)
    context = args.get("context", None) 
    response1 = args.get("response1", None) 
    response2 = args.get("response2", None) 

    # Stage 1: Create fact-checking prompt
    factcheck_prompt = format_factcheck_stage_prompt(context, response1, response2)
    
    # Create message log for fact-checking stage
    message_log = []
    user_message = {
        "role": "user",
        "content": factcheck_prompt,
    }
    
    # Apply chat template
    message: list[str] = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    message_log.append(user_message)
    
    # Calculate total length
    total_length = sum(len(msg["token_ids"]) for msg in message_log)
    
    # Check if we need to truncate
    loss_multiplier = 1.0
    if total_length > max_seq_length:
        for msg in message_log:
            msg["token_ids"] = msg["token_ids"][:min(4, max_seq_length // len(message_log))]
        loss_multiplier = 0.0
    
    # Prepare metadata for two-stage environment
    metadata: TwoStageMetadata = {
        "num_responses": num_responses,
        "helpfulness_1": helpfulness_1,
        "helpfulness_2": helpfulness_2,
        "preference_ranking": preference_ranking,
        "factcheck_stage_complete": False,  # Start with fact-checking
        "factcheck_results": None,
    }
    
    # Store context and responses for scoring stage
    metadata["context"] = context
    metadata["response1"] = response1
    metadata["response2"] = response2

    return DatumSpec(
        message_log=message_log,
        length=total_length,
        extra_env_info=metadata,
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name="two_stage_genrm",
    )


# ========================= TWO-STAGE GENERATION WRAPPER =========================
class TwoStageGenerationWrapper:
    """Wrapper that handles the two-stage generation process."""
    
    def __init__(self, base_generation_interface, tokenizer):
        self.base_generation = base_generation_interface
        self.tokenizer = tokenizer
    
    def generate_two_stage(self, batch: BatchedDataDict) -> BatchedDataDict:
        """Generate responses using two-stage approach."""
        
        # Stage 1: Fact-checking
        stage1_batch = self._prepare_stage1_batch(batch)
        stage1_results = self.base_generation.generate_text(stage1_batch)
        
        # Parse fact-check results and prepare stage 2
        print("example results of fact checking: ", stage1_results.get("texts", [])[0] if stage1_results.get("texts") else "NO TEXTS")
        stage2_batch = self._prepare_stage2_batch(batch, stage1_results)
        stage2_results = self.base_generation.generate_text(stage2_batch)
        
        # Combine results
        return self._combine_stage_results(batch, stage1_results, stage2_results)
    
    def _prepare_stage1_batch(self, batch: BatchedDataDict) -> BatchedDataDict:
        """Prepare batch for fact-checking stage."""
        # Extract prompts from message logs for the base generation interface
        prompts = []
        for message_log in batch["message_log"]:
            # Extract the user message content (fact-checking prompt)
            if message_log and len(message_log) > 0 and message_log[0]["role"] == "user":
                content = message_log[0]["content"]
                prompts.append(content)
            else:
                prompts.append("")  # Fallback for malformed message logs
                print(f"[WARNING] Empty or invalid message_log structure in stage 1")
        
        print(f"[DEBUG] Stage 1 - First prompt (truncated): {prompts[0][:200] if prompts else 'NO PROMPTS'}...")
        return BatchedDataDict({"prompts": prompts})
    
    def _prepare_stage2_batch(self, original_batch: BatchedDataDict, stage1_results: BatchedDataDict) -> BatchedDataDict:
        """Prepare batch for scoring stage using fact-check results."""
        
        stage2_prompts = []
        stage1_texts = stage1_results.get("texts", [])
        
        for i, (metadata, factcheck_response) in enumerate(zip(
            original_batch["extra_env_info"], 
            stage1_texts
        )):
            # Extract stored context and responses
            context = metadata.get("context")
            response1 = metadata.get("response1")
            response2 = metadata.get("response2")
            
            # Create scoring stage prompt
            scoring_prompt = format_scoring_stage_prompt(
                context, response1, response2, factcheck_response
            )
            
            stage2_prompts.append(scoring_prompt)
        
        print(f"[DEBUG] Stage 2 - First prompt (truncated): {stage2_prompts[0][:200] if stage2_prompts else 'NO PROMPTS'}...")
        return BatchedDataDict({"prompts": stage2_prompts})
    
    def _combine_stage_results(self, original_batch: BatchedDataDict, stage1_results: BatchedDataDict, stage2_results: BatchedDataDict) -> BatchedDataDict:
        """Combine results from both stages."""
        
        # Update metadata to include fact-check results
        updated_metadata = []
        stage1_texts = stage1_results.get("texts", [])
        
        for i, metadata in enumerate(original_batch["extra_env_info"]):
            updated_meta = metadata.copy()
            updated_meta["factcheck_stage_complete"] = True
            updated_meta["factcheck_results"] = stage1_texts[i] if i < len(stage1_texts) else ""
            updated_metadata.append(updated_meta)
        
        # Return batch with scoring stage results and updated metadata
        result_batch = original_batch.copy()
        result_batch["extra_env_info"] = updated_metadata
        result_batch["generated_texts"] = stage2_results.get("texts", [])
        
        return result_batch


# ========================= TRAINING INTEGRATION =========================

def setup_two_stage_training(config, tokenizer, dataset, val_dataset):
    """Setup two-stage GenRM training pipeline."""
    
    # Setup base components
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
    
    # Wrap generation interface for two-stage processing
    if policy_generation:
        two_stage_generation = TwoStageGenerationWrapper(policy_generation, tokenizer)
        # Replace the generation interface
        policy_generation.generate_text = two_stage_generation.generate_two_stage
    else:
        print("Fail to replace the generation interface to two_stage_generation")
        
    # Setup two-stage environment
    from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
    
    two_stage_env_config = {
        "format_penalty": -100,
        "factcheck_bonus_multiplier": 0.0,  # Adjust as needed
    }
    
    two_stage_env = TwoStageFactCheckEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.genrm_environment_w_fact.TwoStageFactCheckEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(two_stage_env_config)
    
    # Update task to environment mapping
    task_to_env = defaultdict(lambda: two_stage_env)
    task_to_env["two_stage_genrm"] = two_stage_env
    
    val_task_to_env = defaultdict(lambda: two_stage_env)
    val_task_to_env["two_stage_genrm"] = two_stage_env
    
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
        task_to_env,
        val_task_to_env,
    )

# ========================= DATASET SETUP =========================

class TwoStageHelpSteer3Dataset:
    """HelpSteer3 dataset adapted for two-stage processing."""
    
    def __init__(self, data_path: str, split: str = "train", shuffle_seed: int = -1):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Add task name for two-stage processing
                item["task_name"] = "two_stage_genrm"
                self.data.append(item)
        
        self.split = split
        
        if shuffle_seed != -1:
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(self.data)
            print(f"Shuffled {split} dataset with {len(self.data)} samples using seed {shuffle_seed}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def setup_two_stage_data(tokenizer, data_config, env_configs):
    """Setup data for two-stage GenRM training."""
    
    print("\n▶ Setting up two-stage data...")
    
    # Create task spec for two-stage GenRM
    two_stage_task_spec = TaskDataSpec(
        task_name="two_stage_genrm",
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
    )
    
    # Load datasets
    train_data_path = data_config.get("train_data_path")
    val_data_path = data_config.get("val_data_path")
    
    train_dataset = TwoStageHelpSteer3Dataset(
        train_data_path, 
        split="train", 
        shuffle_seed=data_config.get("shuffle_seed_for_training", -1)
    )
    
    val_dataset = None
    if val_data_path:
        val_dataset = TwoStageHelpSteer3Dataset(val_data_path, split="validation")
    
    # Setup task data processors
    task_data_processors = defaultdict(lambda: (two_stage_task_spec, two_stage_genrm_data_processor))
    task_data_processors["two_stage_genrm"] = (two_stage_task_spec, two_stage_genrm_data_processor)
    
    # Create processed datasets
    processed_train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        two_stage_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    
    processed_val_dataset = None
    if val_dataset:
        processed_val_dataset = AllTaskProcessedDataset(
            val_dataset,
            tokenizer,
            two_stage_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
    

    
    return processed_train_dataset, processed_val_dataset

# ========================= MAIN TRAINING SCRIPT =========================


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training for GenRM")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_genrm_w_fact.yaml"
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

    # Setup data
    train_dataset, val_dataset = setup_two_stage_data(
        tokenizer, 
        config["data"], 
        config["env"],
    )
    
    # Setup two-stage training
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
        task_to_env,
        val_task_to_env,
    ) = setup_two_stage_training(config, tokenizer, train_dataset, val_dataset)
    
    # Debug: Print first example
    if len(train_dataset) > 0:
        print("\n[DEBUG] First two-stage example:")
        first_example = train_dataset[0]
        print(f"  Task name: {first_example.get('task_name')}")
        print(f"  Factcheck stage complete: {first_example['extra_env_info'].get('factcheck_stage_complete')}")
        print(f"  Num responses: {first_example['extra_env_info'].get('num_responses')}")
    
    # Run GRPO training with two-stage fact-checking
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

# ========================= EVALUATION SCRIPT =========================

def evaluate_two_stage_genrm(vllm_generation, dataloader, output_file):
    """Evaluate two-stage GenRM system."""
    
    print("\n▶ Running two-stage GenRM evaluation...")
    
    results = []
    two_stage_wrapper = TwoStageGenerationWrapper(vllm_generation, vllm_generation.tokenizer)
    
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        print(f"Processing batch {batch_idx + 1}")
        
        # Run two-stage generation
        try:
            combined_results = two_stage_wrapper.generate_two_stage(batch)
            generated_texts = combined_results.get("generated_texts", [])
            factcheck_results = [meta.get("factcheck_results", "") for meta in combined_results["extra_env_info"]]
            
        except Exception as e:
            print(f"[ERROR] Two-stage generation failed: {e}")
            generated_texts = [""] * len(batch["message_log"])
            factcheck_results = [""] * len(batch["message_log"])
        
        # Process results
        for idx, (final_output, factcheck_output, metadata) in enumerate(zip(
            generated_texts, factcheck_results, batch["extra_env_info"]
        )):
            result = {
                "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                "factcheck_output": factcheck_output,
                "final_output": final_output,
                "metadata": metadata,
                "two_stage_complete": bool(factcheck_output and final_output),
            }
            
            # Parse final scoring output
            try:
                is_valid, scores, ranking, error = parse_scoring_response(
                    final_output, metadata.get("num_responses", 1)
                )
                
                if is_valid:
                    result["predicted_scores"] = [int(s) for s in scores]
                    if ranking:
                        result["predicted_ranking"] = int(ranking)
                else:
                    result["parse_error"] = error
                    
            except Exception as e:
                result["parse_error"] = str(e)
            
            results.append(result)
    
    # Save results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Two-stage results saved to {output_file}")
    
    # Calculate metrics
    calculate_two_stage_metrics(results)

def calculate_two_stage_metrics(results):
    """Calculate metrics for two-stage evaluation."""
    
    total_samples = len(results)
    completed_two_stage = sum(1 for r in results if r.get("two_stage_complete", False))
    
    correct_rankings = 0
    total_rankings = 0
    
    for result in results:
        if result.get("two_stage_complete", False) and "predicted_ranking" in result:
            total_rankings += 1
            true_pref = result["metadata"].get("preference_ranking")
            if true_pref is not None:
                pred_pref = 0 if result["predicted_ranking"] <= 3 else 1
                if pred_pref == true_pref:
                    correct_rankings += 1
    
    print(f"\n📊 Two-Stage GenRM Metrics:")
    print(f"  • Total samples: {total_samples}")
    print(f"  • Two-stage completion rate: {completed_two_stage/total_samples:.2%}")
    if total_rankings > 0:
        print(f"  • Ranking accuracy: {correct_rankings/total_rankings:.2%} ({correct_rankings}/{total_rankings})")
    else:
        print(f"  • No valid rankings found")

if __name__ == "__main__":
    main()