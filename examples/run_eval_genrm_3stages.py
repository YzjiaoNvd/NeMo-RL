# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import argparse
import json
import os
import pprint
from typing import Any, Optional, TypedDict

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.datasets import AllTaskProcessedDataset, eval_collate_fn
from nemo_rl.data.hf_datasets.reward_benchmarks import (
    JudgeBenchDataset,
    RMBenchDataset,
    RewardBenchDataset,
    RewardBench2Dataset,
    HelpSteer3LocalDataset,
    RMBDataset,
)
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides

# Import three-stage environment components
from nemo_rl.environments.genrm_environment_3stages import (
    format_vanilla_genrm_prompt,
    format_factcheck_prompt,
    format_enhanced_genrm_prompt,
    parse_genrm_response,
    parse_factcheck_response,
    calculate_reward_from_scores,
)

class GenRMEvalConfig(TypedDict):
    dataset_name: str
    batch_size: int
    seed: int
    output_file: str
    use_three_stage: bool

class MasterConfig(TypedDict):
    generation: dict
    data: dict
    eval: GenRMEvalConfig
    tokenizer: dict
    cluster: dict

def genrm_eval_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process evaluation data for three-stage GenRM format."""
    prompt = datum_dict.get("prompt", "")
    
    # Tokenize the prompt
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length)
    token_ids = tokenized["input_ids"][0]
    
    # Create message log
    message_log = [{
        "role": "user",
        "content": prompt,
        "token_ids": token_ids,
    }]
    
    # Extract metadata for three-stage evaluation
    metadata = datum_dict.copy()

    return DatumSpec(
        message_log=message_log,
        length=len(token_ids),
        extra_env_info=metadata,
        loss_multiplier=1.0,
        idx=idx,
        task_name="genrm_eval",
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Run GenRM three-stage evaluation")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, 
                       choices=["judgebench", "rmbench", "rewardbench2", "hs3local", "rewardbench", "rmb"],
                       default=None, help="Dataset to evaluate on")
    args, overrides = parser.parse_known_args()
    return args, overrides

def setup_data(tokenizer, data_config, dataset_name):
    """Set up evaluation dataset."""
    print(f"Setting up {dataset_name} dataset...")
    
    # Load dataset based on type
    dataset_loaders = {
        "judgebench": JudgeBenchDataset,
        "rmbench": RMBenchDataset,
        "rewardbench": RewardBenchDataset,
        "rewardbench2": RewardBench2Dataset,
        "hs3local": HelpSteer3LocalDataset,
    }
    
    if dataset_name not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_loader = dataset_loaders[dataset_name]()
    test_dataset = dataset_loader.formatted_ds
    
    if test_dataset is None or len(test_dataset) == 0:
        raise ValueError(f"{dataset_name} dataset is empty or failed to load")
    
    print(f"Loaded {len(test_dataset)} examples")
    
    # Create task spec and processed dataset
    eval_task_spec = TaskDataSpec(task_name="genrm_eval")
    processed_dataset = AllTaskProcessedDataset(
        test_dataset,
        tokenizer,
        eval_task_spec,
        genrm_eval_data_processor,
        max_seq_length=data_config["max_input_seq_length"],
    )
    
    return processed_dataset, dataset_loader

def run_three_stage_evaluation(vllm_generation, dataloader, output_file):
    """Run three-stage evaluation: Vanilla GenRM → Fact-check → Enhanced GenRM."""
    results = []
    
    print("Running three-stage evaluation...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        try:
            batch_results = []
            
            for idx in range(len(batch["message_log"])):
                metadata = batch["extra_env_info"][idx]
                context = metadata.get("context", "")
                response1 = metadata.get("response1", "")
                response2 = metadata.get("response2", "")
                num_responses = metadata.get("num_responses", 2)
                
                result = {
                    "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                    "metadata": metadata,
                }
                
                # STAGE 1: Vanilla GenRM
                vanilla_prompt = format_vanilla_genrm_prompt(context, response1, response2)
                stage1_inputs = BatchedDataDict({"prompts": [vanilla_prompt]})
                stage1_outputs = vllm_generation.generate_text(stage1_inputs)
                vanilla_response = stage1_outputs.get("texts", [""])[0]
                
                result["vanilla_response"] = vanilla_response
                
                # Parse vanilla response
                vanilla_valid, vanilla_scores, vanilla_ranking, vanilla_error = parse_genrm_response(
                    vanilla_response, num_responses
                )
                result["vanilla_parse_success"] = vanilla_valid
                if vanilla_valid:
                    result["vanilla_scores"] = [int(s) for s in vanilla_scores]
                    result["vanilla_ranking"] = int(vanilla_ranking) if vanilla_ranking else None
                else:
                    result["vanilla_error"] = vanilla_error
                
                # STAGE 2: Fact-checking
                factcheck_prompt = format_factcheck_prompt(context, response1, response2)
                stage2_inputs = BatchedDataDict({"prompts": [factcheck_prompt]})
                stage2_outputs = vllm_generation.generate_text(stage2_inputs)
                factcheck_response = stage2_outputs.get("texts", [""])[0]
                
                result["factcheck_response"] = factcheck_response
                
                # Parse fact-checking response
                factcheck_valid, factcheck_results = parse_factcheck_response(
                    factcheck_response, num_responses
                )
                result["factcheck_parse_success"] = factcheck_valid
                result["factcheck_results"] = factcheck_results
                
                # STAGE 3: Enhanced GenRM
                enhanced_prompt = format_enhanced_genrm_prompt(
                    context, response1, response2, factcheck_results
                )
                stage3_inputs = BatchedDataDict({"prompts": [enhanced_prompt]})
                stage3_outputs = vllm_generation.generate_text(stage3_inputs)
                enhanced_response = stage3_outputs.get("texts", [""])[0]
                
                result["enhanced_response"] = enhanced_response
                
                # Parse enhanced response
                enhanced_valid, enhanced_scores, enhanced_ranking, enhanced_error = parse_genrm_response(
                    enhanced_response, num_responses
                )
                result["enhanced_parse_success"] = enhanced_valid
                if enhanced_valid:
                    result["enhanced_scores"] = [int(s) for s in enhanced_scores]
                    result["enhanced_ranking"] = int(enhanced_ranking) if enhanced_ranking else None
                else:
                    result["enhanced_error"] = enhanced_error
                
                # Calculate rewards if both stages parsed successfully
                if vanilla_valid and enhanced_valid:
                    try:
                        # Create dummy metadata for reward calculation
                        reward_metadata = {
                            "num_responses": num_responses,
                            "helpfulness_1": metadata.get("helpfulness_1"),
                            "helpfulness_2": metadata.get("helpfulness_2"),
                            "preference": metadata.get("preference"),
                        }
                        
                        # Calculate initial reward from vanilla scores
                        vanilla_scores_str = [str(s) for s in result["vanilla_scores"]]
                        vanilla_ranking_str = str(result["vanilla_ranking"]) if result.get("vanilla_ranking") else None
                        initial_reward = calculate_reward_from_scores(vanilla_scores_str, vanilla_ranking_str, reward_metadata)
                        
                        # Calculate enhanced reward from enhanced scores
                        enhanced_scores_str = [str(s) for s in result["enhanced_scores"]]
                        enhanced_ranking_str = str(result["enhanced_ranking"]) if result.get("enhanced_ranking") else None
                        enhanced_reward = calculate_reward_from_scores(enhanced_scores_str, enhanced_ranking_str, reward_metadata)
                        
                        # Calculate fact-check bonus and final reward
                        fact_check_bonus = enhanced_reward - initial_reward
                        final_reward = enhanced_reward + fact_check_bonus  # Base + bonus
                        
                        result["initial_reward"] = initial_reward
                        result["enhanced_reward"] = enhanced_reward
                        result["fact_check_bonus"] = fact_check_bonus
                        result["final_reward"] = final_reward
                        
                    except Exception as e:
                        result["reward_calculation_error"] = str(e)
                
                batch_results.append(result)
            
            results.extend(batch_results)
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # Create fallback results for this batch
            for idx in range(len(batch["message_log"])):
                results.append({
                    "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                    "error": str(e),
                    "vanilla_parse_success": False,
                    "enhanced_parse_success": False,
                })
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    calculate_metrics(results)

def calculate_metrics(results):
    """Calculate evaluation metrics for three-stage system."""
    total_samples = len(results)
    vanilla_successful = sum(1 for r in results if r.get("vanilla_parse_success", False))
    enhanced_successful = sum(1 for r in results if r.get("enhanced_parse_success", False))
    
    # Calculate accuracy metrics for samples with valid rewards
    valid_results = [r for r in results if "final_reward" in r]
    
    if valid_results:
        # Vanilla vs Enhanced comparison
        vanilla_rewards = [r["initial_reward"] for r in valid_results]
        enhanced_rewards = [r["enhanced_reward"] for r in valid_results]
        fact_check_bonuses = [r["fact_check_bonus"] for r in valid_results]
        final_rewards = [r["final_reward"] for r in valid_results]
        
        # Calculate improvement metrics
        improved_samples = sum(1 for bonus in fact_check_bonuses if bonus > 0)
        degraded_samples = sum(1 for bonus in fact_check_bonuses if bonus < 0)
        unchanged_samples = sum(1 for bonus in fact_check_bonuses if bonus == 0)
        
        # Calculate accuracy rates
        vanilla_perfect = sum(1 for r in vanilla_rewards if r == 0)
        enhanced_perfect = sum(1 for r in enhanced_rewards if r == 0)
        
        print(f"\nThree-Stage Evaluation Metrics:")
        print(f"  Total samples: {total_samples}")
        print(f"  Vanilla successful parses: {vanilla_successful} ({vanilla_successful/total_samples:.2%})")
        print(f"  Enhanced successful parses: {enhanced_successful} ({enhanced_successful/total_samples:.2%})")
        print(f"  Valid reward calculations: {len(valid_results)} ({len(valid_results)/total_samples:.2%})")
        print(f"")
        print(f"  Fact-checking impact:")
        print(f"    Improved: {improved_samples} ({improved_samples/len(valid_results):.2%})")
        print(f"    Degraded: {degraded_samples} ({degraded_samples/len(valid_results):.2%})")
        print(f"    Unchanged: {unchanged_samples} ({unchanged_samples/len(valid_results):.2%})")
        print(f"")
        print(f"  Accuracy (perfect predictions):")
        print(f"    Vanilla GenRM: {vanilla_perfect}/{len(valid_results)} ({vanilla_perfect/len(valid_results):.2%})")
        print(f"    Enhanced GenRM: {enhanced_perfect}/{len(valid_results)} ({enhanced_perfect/len(valid_results):.2%})")
        print(f"")
        print(f"  Average rewards:")
        print(f"    Vanilla: {np.mean(vanilla_rewards):.2f}")
        print(f"    Enhanced: {np.mean(enhanced_rewards):.2f}")
        print(f"    Fact-check bonus: {np.mean(fact_check_bonuses):.2f}")
        print(f"    Final reward: {np.mean(final_rewards):.2f}")
    else:
        print(f"\nNo valid reward calculations found")

def main():
    args, overrides = parse_args()
    
    # Load configuration
    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "genrm_eval_3stages.yaml")
    
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Apply overrides
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    
    # Override dataset if specified
    if args.dataset:
        config["eval"]["dataset_name"] = args.dataset
    
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    
    print("Final config:")
    pprint.pprint(config)
    
    # Initialize
    set_seed(config["eval"]["seed"])
    init_ray()
    
    # Setup tokenizer and generation config
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=True
    )
    
    # Setup data
    dataset, _ = setup_data(tokenizer, config["data"], config["eval"]["dataset_name"])
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    
    # Setup cluster and vLLM
    print("Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="genrm_eval_cluster",
        bundle_ct_per_node_list=[config["cluster"]["gpus_per_node"]] * config["cluster"]["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=config["cluster"]["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    
    print("Setting up vLLM generation...")
    vllm_generation = VllmGeneration(cluster=cluster, config=config["generation"])
    vllm_generation.prepare_for_generation()
    
    try:
        # Run evaluation
        run_three_stage_evaluation(vllm_generation, dataloader, config["eval"]["output_file"])
    finally:
        # Cleanup
        vllm_generation.finish_generation()
        vllm_generation.shutdown()

if __name__ == "__main__":
    import numpy as np
    main()