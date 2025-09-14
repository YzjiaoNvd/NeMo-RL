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

# Import two-stage environment components for fact2
from nemo_rl.environments.genrm_environment_w_fact2 import (
    format_factcheck_stage_prompt,
    parse_scoring_response,
    parse_fact_checking_response,
    filter_wrong_and_unknown_facts,
)


class GenRMEvalConfig(TypedDict):
    dataset_name: str
    batch_size: int
    seed: int
    output_file: str
    use_two_stage: bool


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
    """Process evaluation data for GenRM format with fact2 environment."""
    # Extract data
    context = datum_dict.get("context", "")
    response1 = datum_dict.get("response1", "")
    response2 = datum_dict.get("response2", "")
    
    # For fact2 GRPO, we always start with the fact-checking stage
    # The environment will handle the transition to scoring stage with filtering
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
    user_message["content"] = message  
    token_length = len(user_message["token_ids"])

    message_log.append(user_message)

    # Extract metadata for two-stage evaluation
    metadata = datum_dict.copy()
    

    # Debug: Print extracted metadata
    if idx < 3:
        print(f"  Extracted metadata: {metadata}")

    return DatumSpec(
        message_log=message_log,
        length=token_length,
        extra_env_info=metadata,
        loss_multiplier=1.0,
        idx=idx,
        task_name="genrm_eval",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run GenRM two-stage evaluation with fact2 environment")
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
        "rmb": RMBDataset,
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


def create_filtered_scoring_prompt(context: str, response1: str, response2: str, factcheck_results: str) -> str:
    """
    Create scoring prompt with filtered fact-checking results (Wrong/Unknown facts only).
    This mimics what the fact2 environment does internally.
    """
    # Filter to only include Wrong and Unknown facts
    filtered_response = filter_wrong_and_unknown_facts(factcheck_results)
    
    scoring_prompt = f"""You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.
In the previous conversation, there are two model responses and the conversation context from the User. 
You need to refer to the [Helpfulness Scoring Guidelines] to score two response with two individual scores and a ranking score based on the [Ranking Scoring Guidelines]
Before scoring, please refer to the following fact-checking analysis that identifies ONLY problematic facts and analyze step by step. Your scoring needs to be as strict as possible. Please strictly follow the required output format.

**Fact-Checking Results (Wrong/Unknown facts only):**
{filtered_response}

[Helpfulness Scoring Guidelines]

When evaluating Helpfulness, consider the following factors:

- Correctness/Completeness: Is the response accurate and complete?
- Coherence/Clarity: Is the response clear, coherent, and easy to understand?
- Instruction following: Does the response follow the instructions and fulfill the user's request?
- Relevance: Is the response relevant to the user's query/input?
- Level of Detail and Creativity: Does the response provide enough detail without being too verbose? Does it show creativity but not hallucinations?

**Score 5: Extremely Helpful**
- The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for.
- It accurately acts on the user's request, without unnecessary information.
- If a user request is not possible/in line with desired model behavior, a helpful response provides useful context and rationale.

**Score 4: Mostly Helpful**
- The response is mostly helpful and mainly aligned with what the user was looking for.
- There is still some room for improvement, but the response is generally useful.

**Score 3: Partially Helpful**
- The response is partially helpful but misses the overall goal of the user's query/input in some way.
- The response did not fully satisfy what the user was looking for.

**Score 2: Borderline Unhelpful**
- The response is borderline unhelpful and mostly does not capture what the user was looking for.
- However, it is still usable and helpful in a small way.

**Score 1: Not Helpful**
- The response is not useful or helpful at all.
- The response completely missed the essence of what the user wanted.

[Ranking Scoring Guidelines]

Ranking score is used to rank the two responses based on their helpfulness. Even if you give the same individual helpfulness score for both responses, you need to differentiate them strictly.
The ranking score is a number between 1 and 6, where:
1 = Response 1 is much better than Response 2
2 = Response 1 is better than Response 2  
3 = Response 1 is slightly better than Response 2
4 = Response 2 is slightly better than Response 1
5 = Response 2 is better than Response 1
6 = Response 2 is much better than Response 1

#### Output Format Requirements ####
[The Begin of Analysis on Response 1]
Analysis on response 1 (considering the fact-checking results above)
[The End of Analysis on Response 1]

[The Begin of Analysis on Response 2]
Analysis on response 2 (considering the fact-checking results above)
[The End of Analysis on Response 2]

[The Begin of Individual Scores]
\\boxed{{x, y}} (the scores of each response in order)
[The End of Individual Scores]

[The Begin of Ranking Score]
\\boxed{{z}} 
[The End of Ranking Score]"""
    
    return scoring_prompt


def run_two_stage_evaluation(vllm_generation, dataloader, tokenizer, output_file):
    """Run two-stage evaluation with fact2 approach: fact-checking then filtered scoring."""
    results = []
    
    print("Running two-stage evaluation with fact2 environment (filtered approach)...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        try:
            # STAGE 1: Fact-checking
            factcheck_prompts = []
            for metadata in batch["extra_env_info"]:
                context = metadata.get("context", "")
                response1 = metadata.get("response1", "")
                response2 = metadata.get("response2", "")
                
                factcheck_prompt = format_factcheck_stage_prompt(context, response1, response2)
                user_message = {
                    "role": "user",
                    "content": factcheck_prompt,
                }
                factcheck_message = tokenizer.apply_chat_template(  # type: ignore
                    [user_message],
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )
                factcheck_prompts.append(factcheck_message)
            
            # Collect the factcheck results
            stage1_inputs = BatchedDataDict({"prompts": factcheck_prompts})
            stage1_outputs = vllm_generation.generate_text(stage1_inputs)
            factcheck_responses = stage1_outputs.get("texts", [""] * len(factcheck_prompts))

            updated_factcheck_responses = []
            for factcheck_response in factcheck_responses:
                updated_factcheck_responses.append(factcheck_response + "\n" + tokenizer.eos_token + "\n")

            # STAGE 2: Scoring with filtered fact-checking results (fact2 approach)
            scoring_prompts = []
            for i, metadata in enumerate(batch["extra_env_info"]):
                context = metadata.get("context", "")
                response1 = metadata.get("response1", "")
                response2 = metadata.get("response2", "")
                
                factcheck_result = factcheck_responses[i]
                # Create scoring prompt with filtered fact-checking results (fact2 approach)
                scoring_prompt = create_filtered_scoring_prompt(
                    context, response1, response2, factcheck_result
                )
                user_message = {
                    "role": "user",
                    "content": scoring_prompt,
                }
                scoring_message = tokenizer.apply_chat_template(  # type: ignore
                    [user_message],
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )
                scoring_prompts.append(scoring_message)
            
            # Generate scoring responses
            two_stage_prompts = []
            for factcheck_prompt, updated_factcheck_response, scoring_prompt in zip(factcheck_prompts, updated_factcheck_responses, scoring_prompts):
                two_stage_prompts.append(factcheck_prompt + updated_factcheck_response + scoring_prompt)
            
            if batch_idx == 0:  # Debug first batch
                print("Sample two_stage_prompt (fact2 filtered approach):")
                print(two_stage_prompts[0][:1000] + "...")
            
            stage2_inputs = BatchedDataDict({"prompts": two_stage_prompts})
            stage2_outputs = vllm_generation.generate_text(stage2_inputs)
            scoring_responses = stage2_outputs.get("texts", [""] * len(scoring_prompts))
            
            # Process results
            for idx, (factcheck_resp, scoring_resp, metadata) in enumerate(zip(
                factcheck_responses, scoring_responses, batch["extra_env_info"]
            )):
                result = {
                    "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                    "factcheck_response": factcheck_resp,
                    "scoring_response": scoring_resp,
                    "metadata": metadata,
                    "approach": "filtered_two_stage_fact2",  # Mark as fact2 approach
                }
                
                # Parse scoring response for metrics
                is_valid, scores, ranking, error_msg = parse_scoring_response(
                    scoring_resp, metadata.get("num_responses", 2)
                )
                
                result["scoring_parse_success"] = is_valid
                if is_valid:
                    result["predicted_scores"] = [int(s) for s in scores]
                    if ranking:
                        result["predicted_ranking"] = int(ranking)
                else:
                    result["scoring_parse_error"] = error_msg
                
                # Store filtered fact-checking results for analysis
                if factcheck_resp:
                    is_parsed, parsed_response = parse_fact_checking_response(factcheck_resp)
                    if is_parsed:
                        filtered_facts = filter_wrong_and_unknown_facts(parsed_response)
                        result["filtered_factcheck_results"] = filtered_facts
                        result["original_factcheck_parsed"] = True
                    else:
                        result["original_factcheck_parsed"] = False
                
                results.append(result)
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # Create fallback results
            for idx in range(len(batch["message_log"])):
                results.append({
                    "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                    "error": str(e),
                    "scoring_parse_success": False,
                    "metadata": metadata,
                })
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    calculate_metrics(results)


def calculate_metrics(results):
    """Calculate evaluation metrics for fact2 approach."""
    total_samples = len(results)
    successful_parses = sum(1 for r in results if r.get("scoring_parse_success", False))
    successful_factcheck_parses = sum(1 for r in results if r.get("original_factcheck_parsed", False))
    
    # Calculate ranking accuracy
    correct_rankings = 0
    total_rankings = 0
    
    for result in results:
        if result.get("scoring_parse_success", False) and "predicted_ranking" in result:
            total_rankings += 1
            true_pref = result["metadata"].get("preference")
            if true_pref is not None:
                pred_pref = 0 if result["predicted_ranking"] <= 3 else 1
                if pred_pref == true_pref:
                    correct_rankings += 1
    
    print(f"\nEvaluation Metrics (Fact2 Filtered Approach):")
    print(f"  Total samples: {total_samples}")
    print(f"  Successful fact-check parses: {successful_factcheck_parses} ({successful_factcheck_parses/total_samples:.2%})")
    print(f"  Successful scoring parses: {successful_parses} ({successful_parses/total_samples:.2%})")
    if total_rankings > 0:
        print(f"  Ranking accuracy: {correct_rankings/total_rankings:.2%} ({correct_rankings}/{total_rankings})")
    else:
        print(f"  No valid rankings found")
    
    # Additional fact2-specific metrics
    samples_with_filtered_facts = sum(1 for r in results if r.get("filtered_factcheck_results"))
    print(f"  Samples with filtered fact-checking results: {samples_with_filtered_facts}")


def main():
    args, overrides = parse_args()
    
    # Load configuration
    if not args.config:
        # Try to find a fact2-specific config, fallback to fact1 config
        fact2_config = os.path.join(os.path.dirname(__file__), "configs", "genrm_eval_w_fact2.yaml")
        fact1_config = os.path.join(os.path.dirname(__file__), "configs", "genrm_eval_w_fact.yaml")
        
        if os.path.exists(fact2_config):
            args.config = fact2_config
        elif os.path.exists(fact1_config):
            args.config = fact1_config
            print("Using fact1 config as fact2 config not found")
        else:
            raise FileNotFoundError("No suitable config file found")
    
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Apply overrides
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    
    # Override dataset if specified
    if args.dataset:
        config["eval"]["dataset_name"] = args.dataset
    
    # Update output file to indicate fact2 approach
    if "output_file" in config["eval"]:
        base_name = config["eval"]["output_file"]
        if "fact1" in base_name:
            config["eval"]["output_file"] = base_name.replace("fact1", "fact2")
        elif "fact2" not in base_name:
            # Add fact2 suffix
            name_parts = base_name.rsplit(".", 1)
            if len(name_parts) == 2:
                config["eval"]["output_file"] = f"{name_parts[0]}_fact2.{name_parts[1]}"
            else:
                config["eval"]["output_file"] = f"{base_name}_fact2"
    
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
        name="genrm_eval_fact2_cluster",
        bundle_ct_per_node_list=[config["cluster"]["gpus_per_node"]] * config["cluster"]["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=config["cluster"]["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    
    print("Setting up vLLM generation...")
    vllm_generation = VllmGeneration(cluster=cluster, config=config["generation"])
    vllm_generation.prepare_for_generation()
    
    try:
        # Run evaluation with fact2 approach
        print("Running evaluation with fact2 filtered two-stage approach...")
        run_two_stage_evaluation(vllm_generation, dataloader, tokenizer, config["eval"]["output_file"])
    finally:
        # Cleanup
        vllm_generation.finish_generation()
        vllm_generation.shutdown()


if __name__ == "__main__":
    main()