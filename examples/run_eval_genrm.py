# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import argparse
import os
import pprint
from typing import Any, Optional, TypedDict

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
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
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
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
        default="judgebench",
        help="Dataset to evaluate on"
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
        outputs = vllm_generation.generate_text(inputs)["texts"]
        
        # Process outputs and compare with ground truth
        for idx, (output, metadata) in enumerate(zip(outputs, batch["extra_env_info"])):
            result = {
                "idx": batch["idx"][idx].item(),
                "prediction": output,
                "metadata": metadata,
            }
            
            # Parse the prediction to extract scores
            try:
                # Extract individual scores
                if "[The Begin of Individual Scores]" in output:
                    scores_section = output.split("[The Begin of Individual Scores]")[1].split("[The End of Individual Scores]")[0]
                    scores_text = scores_section.split("\\boxed{")[1].split("}")[0]
                    scores = [int(s.strip()) for s in scores_text.split(",")]
                    result["predicted_scores"] = scores
                
                # Extract ranking score if present
                if "[The Begin of Ranking Score]" in output:
                    ranking_section = output.split("[The Begin of Ranking Score]")[1].split("[The End of Ranking Score]")[0]
                    ranking_text = ranking_section.split("\\boxed{")[1].split("}")[0]
                    result["predicted_ranking"] = int(ranking_text.strip())
                    
            except Exception as e:
                print(f"Error parsing output for idx {result['idx']}: {e}")
                result["parse_error"] = str(e)
            
            results.append(result)
    
    # Save results
    import json
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
        if "predicted_ranking" in result and result["metadata"].get("preference_ranking") is not None:
            total_rankings += 1
            # Convert preference to expected ranking
            # Assuming preference follows the convention: 1-3 means response 1 is better, 4-6 means response 2 is better
            true_pref = result["metadata"]["preference_ranking"]
            pred_rank = result["predicted_ranking"]
            
            # Check if prediction matches ground truth direction
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
    
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    
    # Print config
    print("Final config:")
    pprint.pprint(config)
    
    # Initialize Ray
    init_ray()
    
    # Setup tokenizer
    tokenizer = get_tokenizer(config["tokenizer"])
    
    # Setup data
    dataset, dataset_loader = setup_data(
        tokenizer, 
        config["data"], 
        config["eval"]["dataset_name"],
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
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
    output_file = config["eval"]["output_file"].format(
        dataset=config["eval"]["dataset_name"]
    )
    evaluate_genrm(vllm_generation, dataloader, output_file)
    
    # Cleanup
    vllm_generation.shutdown()


if __name__ == "__main__":
    main()