# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import argparse
import json
import os
import pprint
import traceback
from typing import Any, Optional, TypedDict

import torch
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
    RewardBenchDataset,
)
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster, init_ray
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.models.generation import configure_generation_config

class GenRMEvalConfig(TypedDict):
    dataset_name: str  # "judgebench", "rmbench", or "rewardbench"
    batch_size: int
    seed: int
    output_file: str


class MasterConfig(TypedDict):
    generation: VllmConfig
    data: DataConfig
    eval: GenRMEvalConfig
    tokenizer: dict
    cluster: ClusterConfig


def genrm_eval_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process evaluation data for GenRM format."""
    # Format the prompt according to GenRM template
    prompt = datum_dict["prompt"]
    
    # Create message log
    message_log = []
    user_message = {
        "role": "user",
        "content": prompt,
    }
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    
    # Handle case where apply_chat_template returns a list
    if isinstance(message, list):
        message = message[0]
    
    user_message["token_ids"] = tokenizer(message, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)
    
    # Calculate total length
    total_length = sum(len(msg["token_ids"]) for msg in message_log)
    
    # Check if we need to truncate
    loss_multiplier = 1.0
    if total_length > max_seq_length:
        for msg in message_log:
            msg["token_ids"] = msg["token_ids"][:max_seq_length]
        loss_multiplier = 0.0
    
    # Extract metadata
    metadata = {
        "num_responses": datum_dict.get("num_responses", 2),
        "helpfulness_1": datum_dict.get("label_1", None),
        "helpfulness_2": datum_dict.get("label_2", None),
        "preference_ranking": datum_dict.get("preference", None),
        "ground_truth": datum_dict.get("ground_truth", None),
    }

    return DatumSpec(
        message_log=message_log,
        length=total_length,
        extra_env_info=metadata,
        loss_multiplier=loss_multiplier,
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
        choices=["judgebench", "rmbench", "rewardbench"],
        default=None,
        help="Dataset to evaluate on (overrides config)"
    )
    
    args, overrides = parser.parse_known_args()
    return args, overrides


def setup_data(tokenizer, data_config, dataset_name):
    """Set up evaluation dataset."""
    print(f"\n▶ Setting up {dataset_name} dataset...")
    
    # Load dataset
    if dataset_name == "judgebench":
        dataset_loader = JudgeBenchDataset()
    elif dataset_name == "rmbench":
        dataset_loader = RMBenchDataset()
    elif dataset_name == "rewardbench":
        dataset_loader = RewardBenchDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Check if dataset loaded successfully
    if len(dataset_loader.formatted_ds["test"]) == 0:
        print(f"⚠️ Warning: {dataset_name} dataset is empty or failed to load.")
        print("Creating a dummy example for testing...")
        # Create a dummy dataset for testing
        from datasets import Dataset
        dummy_data = {
            "prompt": ["Test prompt"],
            "num_responses": [2],
            "label_1": [3],
            "label_2": [4],
            "preference": [5],
            "ground_truth": [5],
        }
        dataset_loader.formatted_ds["test"] = Dataset.from_dict(dummy_data)
    
    print(f"  ✓ Loaded {len(dataset_loader.formatted_ds['test'])} examples")
    
    # Create task spec
    eval_task_spec = TaskDataSpec(
        task_name="genrm_eval",
    )
    
    # Create processed dataset
    processed_dataset = AllTaskProcessedDataset(
        dataset_loader.formatted_ds["test"],
        tokenizer,
        eval_task_spec,
        genrm_eval_data_processor,
        max_seq_length=data_config["max_input_seq_length"],
    )
    
    return processed_dataset, dataset_loader


def extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \\boxed{} format."""
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
    """Run evaluation and save results."""
    results = []
    
    print("\n▶ Running evaluation...")
    for batch in tqdm(dataloader):
        # Generate responses
        prompts = []
        for message_log in batch["message_log"]:
            content = [msg["content"] for msg in message_log]
            prompts.append("".join(content))
        
        inputs = BatchedDataDict({"prompts": prompts})
        
        # Add stop_strings if present
        if "stop_strings" in batch:
            inputs["stop_strings"] = batch["stop_strings"]
        else:
            inputs["stop_strings"] = [None] * len(prompts)
        
        outputs = vllm_generation.generate_text(inputs)["texts"]
        
        # Process outputs and compare with ground truth
        for idx, (output, metadata) in enumerate(zip(outputs, batch["extra_env_info"])):
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
    import json
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Calculate metrics
    calculate_metrics(results)


def calculate_metrics(results):
    """Calculate evaluation metrics."""
    correct_rankings = 0
    total_rankings = 0
    score_differences = []
    
    for result in results:
        if "predicted_ranking" in result and result["metadata"].get("preference") is not None:
            total_rankings += 1
            # Convert preference to expected ranking
            # Assuming preference follows the convention: 1-3 means response 1 is better, 4-6 means response 2 is better
            true_pref = result["metadata"]["preference"]
            pred_rank = result["predicted_ranking"]
            
            # For judgebench/rmbench, lower values (1-3) mean first response is better
            # For rewardbench, we just check if the model correctly identified the chosen one
            if isinstance(true_pref, str) and true_pref == "chosen":
                # RewardBench case
                if pred_rank <= 3:  # Model thinks response 1 is better
                    correct_rankings += 1
            elif isinstance(true_pref, (int, float)):
                # JudgeBench/RMBench case
                if (true_pref <= 3 and pred_rank <= 3) or (true_pref > 3 and pred_rank > 3):
                    correct_rankings += 1
    
    if total_rankings > 0:
        accuracy = correct_rankings / total_rankings
        print(f"\n📊 Evaluation Metrics:")
        print(f"  • Ranking Accuracy: {accuracy:.2%} ({correct_rankings}/{total_rankings})")
    else:
        print("\n⚠️ No valid rankings found in results")


def main():
    args, overrides = parse_args()
    
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "genrm_eval.yaml"
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
    
    # Create dataloader
    from nemo_rl.data.datasets import eval_collate_fn
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
    
    # Run evaluation
    output_file = config["eval"]["output_file"]
    evaluate_genrm(vllm_generation, dataloader, output_file)
    
    # Cleanup
    vllm_generation.shutdown()


if __name__ == "__main__":
    main()