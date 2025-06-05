# nemo_rl/data/processors/genrm_processor.py
import json
from typing import Any

import torch

from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec


def helpsteer3_genrm_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process HelpSteer3 data for GenRM training."""
    
    # Extract the system prompt and user input from the data
    system_prompt = datum_dict.get("system", "")
    user_input = datum_dict.get("input", "")
    
    # For GenRM, we need to format the input to ask the model to evaluate responses
    # Based on the training data path, it seems they use a specific format
    if "num_responses" in datum_dict and datum_dict["num_responses"] == 2:
        # Two response evaluation
        response1 = datum_dict.get("response1", "")
        response2 = datum_dict.get("response2", "")
        
        prompt = f"""Please evaluate the following two responses and provide:
1. Individual helpfulness scores (0-5) for each response
2. A preference ranking indicating which response is better (1 if first is better, 2 if second is better)

User Input: {user_input}

Response 1: {response1}

Response 2: {response2}

Please provide your evaluation in the following format:
[The Begin of Individual Scores]
\\boxed{{score1, score2}}
[The End of Individual Scores]

[The Begin of Ranking Score]
\\boxed{{ranking}}
[The End of Ranking Score]"""
    else:
        # Single response evaluation
        response = datum_dict.get("response", "")
        
        prompt = f"""Please evaluate the following response and provide a helpfulness score (0-5).

User Input: {user_input}

Response: {response}

Please provide your evaluation in the following format:
[The Begin of Individual Scores]
\\boxed{{score}}
[The End of Individual Scores]"""
    
    # Create message log
    message_log = []
    
    # Add system message if present
    if system_prompt:
        system_msg = {
            "role": "system",
            "content": system_prompt,
        }
        system_tokens = tokenizer(system_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        system_msg["token_ids"] = system_tokens
        message_log.append(system_msg)
    
    # Add user message with the evaluation prompt
    user_msg = {
        "role": "user",
        "content": prompt,
    }
    user_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    user_msg["token_ids"] = user_tokens
    message_log.append(user_msg)
    
    # Calculate total length
    total_length = sum(len(msg["token_ids"]) for msg in message_log)
    
    # Prepare metadata for environment
    metadata = {
        "num_responses": datum_dict.get("num_responses", 1),
        "helpfulness_1": datum_dict.get("helpfulness_1", None),
        "helpfulness_2": datum_dict.get("helpfulness_2", None),
        "preference_ranking": datum_dict.get("preference_ranking", None),
    }
    
    # Check if we need to truncate
    loss_multiplier = 1.0
    if total_length > max_seq_length:
        # Truncate messages
        for msg in message_log:
            msg["token_ids"] = msg["token_ids"][:max_seq_length // len(message_log)]
        loss_multiplier = 0.0
    
    return DatumSpec(
        message_log=message_log,
        length=total_length,
        extra_env_info=metadata,
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name="genrm",
    )
