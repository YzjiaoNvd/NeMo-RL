# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Optional

from datasets import Dataset, load_dataset, concatenate_datasets
from nemo_rl.data.interfaces import TaskDataSpec
import json
import numpy as np
import torch

# GenRM prompt template
GENRM_PROMPT_TEMPLATE = """You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.
Given the context of the conversation (the last turn is the User's query) and one or two responses from the Assistant, you need to refer to the [Helpfulness Scoring Guidelines] to score each individual response.
If there are two responses, you need to also give a ranking score based on the [Ranking Scoring Guidelines].
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

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

#### Conversation Context ####
{context}

#### Responses to be Scored ####
{responses}

#### Output Format Requirements ####
First give your analysis on each responses in the format of:
[The Begin of Analysis on Response i]
Analysis on the i-th response
[The End of Analysis on Response i]

Then give the scores of each response in order, separate by comma in the boxed, adhering this format:
[The Begin of Individual Scores]
\\boxed{{x, y}} if there exists 2 responses
[The End of Individual Scores]

If there are two responses, give the relative ranking score in the format of:
[The Begin of Ranking Score]
\\boxed{{z}} 
[The End of Ranking Score]
You don't need to give a ranking score if only one response is provided."""


def format_genrm_prompt(context: str, response1: str, response2: Optional[str] = None) -> str:
    """Format the GenRM prompt with context and responses."""
    if response2 is None:
        responses = f"[The Begin of Response 1]\n{response1}\n[The End of Response 1]"
    else:
        responses = (
            f"[The Begin of Response 1]\n{response1}\n[The End of Response 1]\n"
            f"[The Begin of Response 2]\n{response2}\n[The End of Response 2]\n"
        )
    
    return GENRM_PROMPT_TEMPLATE.format(
        context=context,
        responses=responses
    )



def format_judgebench_example(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Format JudgeBench data for GenRM evaluation."""
    # Extract conversation context and responses based on actual JudgeBench format
    context = data.get("question", "")
    response1 = data.get("response_A", "")
    response2 = data.get("response_B", "")
    
    # Parse the label field (e.g., "B>A" means B is better than A)
    label = data.get("label", "")
    preference = None
    if label == "A>B":
        preference = 0  # Response 1 (A) is better
    else: # label == "B>A"
        preference = 1  # Response 2 (B) is better
    
    # JudgeBench doesn't have individual scores, just preference labels
    result1 = {
        "prompt": format_genrm_prompt(context, response1, response2),
        "num_responses": 2,
        "label_1": None,  # JudgeBench doesn't provide individual scores
        "label_2": None,
        "preference": preference,
    }
    
    result2 = {
        "prompt": format_genrm_prompt(context, response2, response1),
        "num_responses": 2,
        "label_1": None,  # JudgeBench doesn't provide individual scores
        "label_2": None,
        "preference": 1-preference,
    }
    
    return [result1, result2]


def format_rmbench_example(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Format RM-Bench data for GenRM evaluation."""
    # Extract prompt and responses
    prompt_text = data.get("prompt", "")
    chosen_responses = data.get("chosen", [])
    rejected_responses = data.get("rejected", [])

    # Format as conversation context
    context = f"User: {prompt_text}"
    
    # Create comparisons between each chosen and rejected response
    examples = []
    for chosen_resp, rejected_resp in zip(chosen_responses, rejected_responses):
        example1 = {
            "prompt": format_genrm_prompt(context, chosen_resp, rejected_resp),
            "num_responses": 2,
            "label_1": None,  # RM-Bench doesn't have individual scores
            "label_2": None,
            "preference": 0,  # 0 = first response (chosen) is better
        }
        examples.append(example1)

        example2 = {
            "prompt": format_genrm_prompt(context, rejected_resp, chosen_resp),
            "num_responses": 2,
            "label_1": None,  # RM-Bench doesn't have individual scores
            "label_2": None,
            "preference": 1,  # 1 = second response (chosen) is better
        }
        examples.append(example2)

    return examples
    


def format_rewardbench2_example(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Format RM-Bench data for GenRM evaluation."""
    # Extract prompt and responses
    prompt_text = data.get("prompt", "")
    chosen_responses = data.get("chosen", [])
    rejected_responses = data.get("rejected", [])

    assert len(chosen_responses) == 1
    assert len(rejected_responses) == 3

    # Format as conversation context
    context = f"User: {prompt_text}"
    
    # Create comparisons between each chosen and rejected response
    examples = []
    for rejected_resp in rejected_responses:
        # Create the GenRM prompt
        prompt = format_genrm_prompt(context, chosen_responses[0], rejected_resp)    
        # Create example with metadata
        example = {
            "prompt": prompt,
            "num_responses": 2,
            "label_1": None,  # RM-Bench doesn't have individual scores
            "label_2": None,
            "preference": 0,  # 0 = first response (chosen) is better
            "ground_truth": None,
        }
        examples.append(example)

    return examples




class JudgeBenchDataset:
    """JudgeBench dataset for GenRM evaluation."""
    def __init__(self):
        # Load both splits
        gpt_ds = load_dataset("ScalerLab/JudgeBench", split="gpt")
        claude_ds = load_dataset("ScalerLab/JudgeBench", split="claude")
        
        # Merge the datasets
        merged_ds = concatenate_datasets([gpt_ds, claude_ds])
        
        # Manually expand the dataset by iterating through each example
        all_formatted_examples = []
        for example in merged_ds:
            # Get the three formatted examples for this sample
            formatted_examples = format_judgebench_example(example)
            all_formatted_examples.extend(formatted_examples)
        
        # Create a new dataset from the expanded examples
        self.formatted_ds = Dataset.from_list(all_formatted_examples)



class RMBenchDataset:
    """RM-Bench dataset for GenRM evaluation."""
    
    def __init__(self):
        # Load both splits
        ds = load_dataset("THU-KEG/RM-Bench", split="train")

        # Manually expand the dataset by iterating through each example
        all_formatted_examples = []
        for example in ds:
            # Get the three formatted examples for this sample
            formatted_examples = format_rmbench_example(example)
            all_formatted_examples.extend(formatted_examples)
        
        # Create a new dataset from the expanded examples
        self.formatted_ds = Dataset.from_list(all_formatted_examples)





class RewardBench2Dataset:
    """RewardBench dataset for GenRM evaluation."""
    
    def __init__(self):
        # Load all splits except Ties
        ds = load_dataset("allenai/reward-bench-2", split="test")
        filter_ds = ds.filter(lambda ex: ex["subset"] != "Ties")

        # Manually expand the dataset by iterating through each example
        all_formatted_examples = []
        for example in filter_ds:
            # Get the three formatted examples for this sample
            formatted_examples = format_rewardbench2_example(example)
            all_formatted_examples.extend(formatted_examples)
        
        # Create a new dataset from the expanded examples
        self.formatted_ds = Dataset.from_list(all_formatted_examples)


######issue to be fixed: change the input file 
class HelpSteer3LocalDataset(torch.utils.data.Dataset):
    """Dataset for loading HelpSteer3 data from local JSONL files."""
    def __init__(self, data_path: str="/lustre/fsw/portfolios/llmservice/users/yizhuj/datasets/hs3_genrm/val_data.jsonl", shuffle_seed: int = -1):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                one = json.loads(line)
                if one["args"]["num_responses"] == 2:
                    preference = 0 if one["args"]["preference_ranking"] <= 3 else 1
                    example = {
                        "prompt": format_genrm_prompt(one["args"]["context"], one["args"]["response1"], one["args"]["response2"]),
                        "num_responses": 2,
                        "label_1": one["args"]["helpfulness_1"],  
                        "label_2": one["args"]["helpfulness_2"],
                        "preference": preference, 
                        "preference_ranking": one["args"]["preference_ranking"],
                        "context": one["args"]["context"],
                        "response1": one["args"]["response1"],
                        "response2": one["args"]["response2"],
                    }
                    data.append(example)

                    example = {
                        "prompt": format_genrm_prompt(one["args"]["context"], one["args"]["response2"], one["args"]["response1"]),
                        "num_responses": 2,
                        "label_1": one["args"]["helpfulness_2"],  
                        "label_2": one["args"]["helpfulness_1"],
                        "preference": 1 - preference, 
                        "preference_ranking": 7 - one["args"]["preference_ranking"],
                        "context": one["args"]["context"],
                        "response1": one["args"]["response2"],
                        "response2": one["args"]["response1"],
                    }
                    data.append(example)
        
        if shuffle_seed != -1:
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(data)
            print(f"Shuffled the dataset with {len(data)} samples using seed {shuffle_seed}")

        self.data = Dataset.from_list(data)
        self.formatted_ds = self.data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Add task_name for compatibility with AllTaskProcessedDataset
        item = self.data[idx].copy()
        item["task_name"] = "genrm"
        return item

