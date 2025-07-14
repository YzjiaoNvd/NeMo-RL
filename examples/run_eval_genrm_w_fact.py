# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Minimal modifications to original evaluation script for two-stage GenRM

import argparse
import json
import os
import pprint
import traceback
from typing import Any, Optional, TypedDict

import torch
import ray
from datasets import Dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, eval_collate_fn
from nemo_rl.data.hf_datasets.reward_benchmarks import (
    JudgeBenchDataset,
    RMBenchDataset,
    RewardBench2Dataset,
    HelpSteer3LocalDataset,
)
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster, init_ray
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.models.generation import configure_generation_config

# Import the two-stage environment (adjust import path as needed)
from nemo_rl.environments.genrm_environment_w_fact import (
    TwoStageFactCheckEnvironment,
    format_factcheck_stage_prompt,
    format_scoring_stage_prompt, 
    parse_scoring_response
)


class GenRMEvalConfig(TypedDict):
    dataset_name: str  # "judgebench", "rmbench", or "rewardbench2"
    batch_size: int
    seed: int
    output_file: str
    use_two_stage: bool  # NEW: Enable two-stage evaluation


class MasterConfig(TypedDict):
    generation: VllmConfig
    data: DataConfig
    eval: GenRMEvalConfig
    tokenizer: dict
    cluster: ClusterConfig
    two_stage_env: dict  # NEW: Two-stage environment config


def genrm_eval_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process evaluation data for GenRM format. (UNCHANGED from original)"""
    # Debug: Print the datum_dict to see what fields are available
    if idx < 3:  # Only print first few examples
        print(f"\n[DEBUG] Example {idx} datum_dict keys: {list(datum_dict.keys())}")
        for key in ["prompt", "num_responses", "label_1", "label_2", "preference", "ground_truth"]:
            if key in datum_dict:
                value = datum_dict[key]
                if isinstance(value, str):
                    print(f"  {key}: {value}...")
                else:
                    print(f"  {key}: {value}")
    
    # The datum_dict already contains the formatted prompt from format_judgebench_example
    prompt = datum_dict.get("prompt", "")
    
    # Tokenize the prompt to get token_ids
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length)
    token_ids = tokenized["input_ids"][0]
    
    # Create message log with tokenized content
    message_log = [
        {
            "role": "user",
            "content": prompt,
            "token_ids": token_ids,
        }
    ]
    
    # Extract metadata - make sure we're getting the actual values
    metadata = {
        "num_responses": datum_dict.get("num_responses", 2),
        "helpfulness_1": datum_dict.get("label_1"),
        "helpfulness_2": datum_dict.get("label_2"),
        "preference_ranking": datum_dict.get("preference"),
        "ground_truth": datum_dict.get("ground_truth"),
    }
    
    # Debug: Print extracted metadata
    if idx < 3:
        print(f"  Extracted metadata: {metadata}")

    return DatumSpec(
        message_log=message_log,
        length=len(token_ids),
        extra_env_info=metadata,
        loss_multiplier=1.0,
        idx=idx,
        task_name="genrm_eval",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run GenRM evaluation")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["judgebench", "rmbench", "rewardbench2", "hs3local"],
        default=None,
        help="Dataset to evaluate on (overrides config)"
    )
    # NEW: Add two-stage option
    parser.add_argument(
        "--two-stage", 
        action="store_true",
        help="Use two-stage fact-checking evaluation"
    )
    
    args, overrides = parser.parse_known_args()
    return args, overrides


def setup_data(tokenizer, data_config, dataset_name):
    """Set up evaluation dataset. (UNCHANGED from original)"""
    print(f"\n▶ Setting up {dataset_name} dataset...")
    
    # Load dataset
    if dataset_name == "judgebench":
        dataset_loader = JudgeBenchDataset()
    elif dataset_name == "rmbench":
        dataset_loader = RMBenchDataset()
    elif dataset_name == "rewardbench2":
        dataset_loader = RewardBench2Dataset()
    elif dataset_name == "hs3local":
        dataset_loader = HelpSteer3LocalDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    test_dataset = dataset_loader.formatted_ds
    if test_dataset is None or len(test_dataset) == 0:
        print(f"⚠️ Warning: {dataset_name} dataset is empty or failed to load.")
        
    
    print(f"  ✓ Loaded {len(test_dataset)} examples")
    
    # Debug: Print first example from the dataset
    if len(test_dataset) > 0:
        print("\n[DEBUG] First example from dataset:")
        first_example = test_dataset[0]
        print(f"  Keys: {list(first_example.keys())}")
        for key, value in first_example.items():
            if isinstance(value, str):
                print(f"  {key}: {value}...")
            else:
                print(f"  {key}: {value}")
    
    # Create task spec
    eval_task_spec = TaskDataSpec(
        task_name="genrm_eval",
    )
    
    # Create processed dataset
    processed_dataset = AllTaskProcessedDataset(
        test_dataset,
        tokenizer,
        eval_task_spec,
        genrm_eval_data_processor,
        max_seq_length=data_config["max_input_seq_length"],
    )
    
    return processed_dataset, dataset_loader


def extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \\boxed{} format. (UNCHANGED from original)"""
    import re
    
    # Try to find \boxed{...} pattern
    pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    
    # Alternative: look for content between curly braces after boxed
    if '\\boxed{' in text:
        start = text.find('\\boxed{') + len('\\boxed{')
        # Find the matching closing brace
        brace_count = 1
        i = start
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            return text[start:i-1].strip()
    
    return None


def evaluate_genrm(vllm_generation, dataloader, output_file):
    """Run evaluation and save results. (UNCHANGED from original)"""
    results = []
    
    print("\n▶ Running evaluation...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # Debug first batch
        if batch_idx == 0:
            print(f"\n[DEBUG] First batch structure:")
            print(f"  Batch keys: {list(batch.keys())}")
            print(f"  Batch size: {len(batch['message_log'])}")
            if len(batch['message_log']) > 0:
                print(f"  First message_log structure: {[msg['role'] for msg in batch['message_log'][0]]}")
                print(f"  First metadata: {batch['extra_env_info'][0]}")
        
        # Generate responses
        prompts = []
        for message_log in batch["message_log"]:
            # Extract just the content from the user message
            if message_log and len(message_log) > 0 and message_log[0]["role"] == "user":
                content = message_log[0]["content"]
                prompts.append(content)
                
                # Debug: Print first prompt
                if batch_idx == 0 and len(prompts) == 1:
                    print(f"\n[DEBUG] First prompt (truncated): {content[:1000]}...")
            else:
                prompts.append("")
                print(f"[WARNING] Empty or invalid message_log structure")
        
        # Create generation input
        inputs = BatchedDataDict({"prompts": prompts})
        
        # Generate using vLLM
        try:
            outputs = vllm_generation.generate_text(inputs)
            generated_texts = outputs.get("texts", [""] * len(prompts))
            
            # Debug first generation
            if batch_idx == 0 and len(generated_texts) > 0:
                print(f"\n[DEBUG] First generation output (truncated): {generated_texts[0][:500]}...")
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            generated_texts = [""] * len(prompts)
        
        # Process outputs and compare with ground truth
        for idx, (output, metadata) in enumerate(zip(generated_texts, batch["extra_env_info"])):
            result = {
                "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                "prediction": output,
                "metadata": metadata,
            }
            
            # Parse the prediction to extract scores
            try:
                # Extract individual scores
                if "[The Begin of Individual Scores]" in output and "[The End of Individual Scores]" in output:
                    scores_section = output.split("[The Begin of Individual Scores]")[1].split("[The End of Individual Scores]")[0]
                    scores_text = extract_boxed_content(scores_section)
                    if scores_text:
                        # Split by comma and convert to integers
                        scores = []
                        for s in scores_text.split(","):
                            s = s.strip()
                            try:
                                scores.append(int(s))
                            except ValueError:
                                # Try to extract just numbers
                                import re
                                num_match = re.search(r'\d+', s)
                                if num_match:
                                    scores.append(int(num_match.group()))
                                else:
                                    scores.append(0)  # Default if can't parse
                        result["predicted_scores"] = scores
                
                # Extract ranking score if present
                if "[The Begin of Ranking Score]" in output and "[The End of Ranking Score]" in output:
                    ranking_section = output.split("[The Begin of Ranking Score]")[1].split("[The End of Ranking Score]")[0]
                    ranking_text = extract_boxed_content(ranking_section)
                    if ranking_text:
                        try:
                            result["predicted_ranking"] = int(ranking_text.strip())
                        except ValueError:
                            # Try to extract just the number
                            import re
                            num_match = re.search(r'\d+', ranking_text)
                            if num_match:
                                result["predicted_ranking"] = int(num_match.group())
                    
            except Exception as e:
                print(f"Error parsing output for idx {result['idx']}: {e}")
                result["parse_error"] = str(e)
            
            results.append(result)
    
    # Save results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Calculate metrics
    calculate_metrics(results)


def evaluate_two_stage_genrm(vllm_generation, dataloader, output_file):
    """NEW: Run two-stage evaluation using generation wrapper."""
    results = []
    
    # Two-stage evaluation using direct generation (imports at top of file)
    # Simplified approach: Use raw fact-checking output directly in scoring stage
    # No parsing of fact-checking responses - just truncate if too long
    
    print("\n▶ Running two-stage evaluation...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # Debug first batch
        if batch_idx == 0:
            print(f"\n[DEBUG] Two-stage first batch structure:")
            print(f"  Batch keys: {list(batch.keys())}")
            print(f"  Batch size: {len(batch['message_log'])}")
            if len(batch['message_log']) > 0:
                print(f"  First message_log structure: {[msg['role'] for msg in batch['message_log'][0]]}")
                print(f"  First metadata: {batch['extra_env_info'][0]}")
        
        try:
            # STAGE 1: Generate fact-checking responses
            print(f"[DEBUG] Starting Stage 1 (fact-checking) for batch {batch_idx}")
            
            # Create fact-checking prompts from metadata (not from message logs which contain GenRM prompts)
            factcheck_prompts = []
            for metadata in batch["extra_env_info"]:
                # Extract context and responses from metadata to create fact-checking prompt
                context = metadata.get("context", "")
                response1 = metadata.get("response1", "")
                response2 = metadata.get("response2", "")
                
                # Create proper fact-checking prompt
                factcheck_prompt = format_factcheck_stage_prompt(context, response1, response2)
                factcheck_prompts.append(factcheck_prompt)
            
            # Generate fact-checking responses
            stage1_inputs = BatchedDataDict({"prompts": factcheck_prompts})
            stage1_outputs = vllm_generation.generate_text(stage1_inputs)
            factcheck_responses = stage1_outputs.get("texts", [""] * len(factcheck_prompts))
            
            # Debug first fact-check response
            if batch_idx == 0 and len(factcheck_responses) > 0:
                print(f"\n[DEBUG] First fact-check response: {factcheck_responses[0]}...")
            
            # STAGE 2: Create scoring prompts with raw fact-check results (no parsing)
            print(f"[DEBUG] Starting Stage 2 (scoring) for batch {batch_idx}")
            
            scoring_prompts = []
            
            for i, (metadata, factcheck_response) in enumerate(zip(
                batch["extra_env_info"], 
                factcheck_responses
            )):
                # Truncate fact-check response if too long (keep first 2000 chars)
                max_factcheck_length = 2000
                truncated_factcheck = factcheck_response
                if len(factcheck_response) > max_factcheck_length:
                    truncated_factcheck = factcheck_response[:max_factcheck_length] + "\n[...truncated]"
                    print(f"[DEBUG] Truncated fact-check response for sample {i} from {len(factcheck_response)} to {len(truncated_factcheck)} chars")
                
                # Extract context and responses for scoring prompt
                context = metadata.get("context", "")
                response1 = metadata.get("response1", "")
                response2 = metadata.get("response2", "")
                
                # Create scoring stage prompt using raw fact-check output
                scoring_prompt = format_scoring_stage_prompt(
                    context, response1, response2, truncated_factcheck
                )
                scoring_prompts.append(scoring_prompt)
            
            # Generate scoring responses
            stage2_inputs = BatchedDataDict({"prompts": scoring_prompts})
            stage2_outputs = vllm_generation.generate_text(stage2_inputs)
            scoring_responses = stage2_outputs.get("texts", [""] * len(scoring_prompts))
            
            # Debug first scoring response
            if batch_idx == 0 and len(scoring_responses) > 0:
                print(f"\n[DEBUG] First scoring response (truncated): {scoring_responses[0][:300]}...")
            
            # Process final results - save raw fact-checking without parsing
            for idx, (factcheck_response, scoring_response, metadata) in enumerate(zip(
                factcheck_responses, scoring_responses, batch["extra_env_info"]
            )):
                result = {
                    "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                    "factcheck_response": factcheck_response,  # Raw fact-checking output
                    "scoring_response": scoring_response,      # Raw scoring output
                    "metadata": metadata,
                    "two_stage_complete": True,
                }
                
                # Parse only the scoring response for metrics (skip fact-check parsing)
                try:
                    is_valid_score, scores, ranking, score_error = parse_scoring_response(
                        scoring_response, metadata.get("num_responses", 2)
                    )
                    
                    if is_valid_score:
                        result["predicted_scores"] = [int(s) for s in scores]
                        if ranking:
                            result["predicted_ranking"] = int(ranking)
                        result["scoring_parse_success"] = True
                    else:
                        result["scoring_parse_error"] = score_error
                        result["scoring_parse_success"] = False
                        
                except Exception as e:
                    result["scoring_parse_error"] = str(e)
                    result["scoring_parse_success"] = False
                
                # Debug first result
                if batch_idx == 0 and idx == 0:
                    print(f"\n[DEBUG] Two-stage first result:")
                    print(f"  Has factcheck_response: {len(result.get('factcheck_response', '')) > 0}")
                    print(f"  Has scoring_response: {len(result.get('scoring_response', '')) > 0}")
                    print(f"  Scoring parse success: {result.get('scoring_parse_success', False)}")
                    print(f"  Has predicted scores: {'predicted_scores' in result}")
                    print(f"  Has predicted ranking: {'predicted_ranking' in result}")
                
                results.append(result)
                
        except Exception as e:
            print(f"[ERROR] Two-stage evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Create fallback results with empty responses
            for idx in range(len(batch["message_log"])):
                result = {
                    "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                    "factcheck_response": "",
                    "scoring_response": "",
                    "metadata": batch["extra_env_info"][idx],
                    "error": str(e),
                    "two_stage_complete": False,
                    "scoring_parse_success": False,
                }
                results.append(result)
    
    # Save results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Two-stage results saved to {output_file}")
    
    # Calculate two-stage metrics
    # calculate_two_stage_metrics(results)



def calculate_metrics(results):
    """Calculate evaluation metrics. (UNCHANGED from original)"""
    correct_rankings = 0
    total_rankings = 0
    
    for result in results:
        total_rankings += 1
        if "predicted_ranking" in result and result["metadata"].get("preference_ranking") is not None:
            # Convert preference to expected ranking
            true_pref = result["metadata"]["preference_ranking"]
            extracted_pred_rank = result["predicted_ranking"]
            pred_rank = 0 if extracted_pred_rank <= 3 else 1
                
            if pred_rank == true_pref:
                correct_rankings += 1
    
    if total_rankings > 0:
        accuracy = correct_rankings / total_rankings
        print(f"\n📊 Evaluation Metrics:")
        print(f"  • Ranking Accuracy: {accuracy:.2%} ({correct_rankings}/{total_rankings})")
    else:
        print("\n⚠️ No valid rankings found in results")

'''
def calculate_two_stage_metrics(results):
    """NEW: Calculate metrics for two-stage evaluation."""
    total_samples = len(results)
    error_samples = sum(1 for r in results if r.get("error", False))
    valid_samples = total_samples - error_samples
    
    if valid_samples == 0:
        print(f"\n⚠️ All {total_samples} samples had errors")
        return
    
    valid_results = [r for r in results if not r.get("error", False)]
    rewards = [r["two_stage_reward"] for r in valid_results]
    mean_reward = sum(rewards) / len(rewards)
    
    print(f"\n📊 Two-Stage Evaluation Metrics:")
    print(f"  • Total Samples: {total_samples}")
    print(f"  • Valid Samples: {valid_samples}")
    print(f"  • Error Rate: {error_samples/total_samples:.2%}")
    print(f"  • Mean Two-Stage Reward: {mean_reward:.2f}")
    
    # Additional quality metrics
    positive_rewards = sum(1 for r in rewards if r > 0)
    print(f"  • Positive Reward Rate: {positive_rewards/valid_samples:.2%}")
'''

def main():
    args, overrides = parse_args()
    
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "genrm_eval_w_fact.yaml"
        )
    
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    
    # Override dataset if specified in command line
    if args.dataset:
        config["eval"]["dataset_name"] = args.dataset
    
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    
    # Print config
    print("Final config:")
    pprint.pprint(config)
    
    # Set seed
    set_seed(config["eval"]["seed"])
    
    # Initialize Ray
    init_ray()

    
    # Setup tokenizer
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=True
    )
    
    # Setup data
    dataset, dataset_loader = setup_data(
        tokenizer, 
        config["data"], 
        config["eval"]["dataset_name"],
    )
    
    # Create dataloader with eval_collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    
    # Setup cluster
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="genrm_eval_cluster",
        bundle_ct_per_node_list=[config["cluster"]["gpus_per_node"]]
        * config["cluster"]["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=config["cluster"]["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    
    # Setup vLLM
    print("\n▶ Setting up vLLM generation...")
    vllm_generation = VllmGeneration(cluster=cluster, config=config["generation"])
    
    # Prepare for generation
    vllm_generation.prepare_for_generation()
    
    # NEW: Choose evaluation method based on config
    output_file = config["eval"]["output_file"]
    use_two_stage = config["eval"].get("use_two_stage", True)
    
    if use_two_stage:
        print("\n▶ Using two-stage evaluation mode...")
        # Run two-stage evaluation using direct generation instead of environment
        evaluate_two_stage_genrm(vllm_generation, dataloader, output_file)
    else:
        print("\n▶ Using standard evaluation mode...")
        # Run standard evaluation (original method)
        evaluate_genrm(vllm_generation, dataloader, output_file)

    # Cleanup
    vllm_generation.finish_generation()
    vllm_generation.shutdown()


if __name__ == "__main__":
    main()