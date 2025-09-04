# Two-Stage General Error Detection GenRM Implementation
import re
import json
import logging
from typing import Any, Optional, TypedDict, Tuple, Dict
import numpy as np
import ray
import torch
from dataclasses import dataclass

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.genrm_environment_base import distance_abs


# ========================= STAGE 1: GENERAL ERROR DETECTION =========================

ERROR_DETECTION_STAGE_PROMPT = """You are an expert error detection analyst. Analyze the given two responses for objective, verifiable errors. Strictly follow the required output format and make your answer as brief as possible.

**Task:** Identify verifiable errors in the responses and provide corrections when possible.

**Error Types to Check:**
- Factual errors (incorrect facts, dates, names, statistics, etc.)
- Mathematical or computational errors (wrong calculations, formulas, numbers)
- Logical contradictions (statements that contradict each other within the response)
- Definitional errors (incorrect definitions of terms, concepts, or processes)
- Citation or reference errors (incorrect sources, misattributed quotes, wrong URLs)
- Code syntax or logical errors (if code is present)
- Temporal/chronological errors (wrong sequences, impossible timelines)
- Formatting violations (if specific format requirements were given)

**Context:** 
{context}

**Responses:**
{responses}

**Output Format:**
[Error Detection for Response 1]
If no errors found, show "No errors identified". Otherwise, list the errors one by one in the following format.
(1) Error Identified: [short sentence or phrase] | Error Type: [one error type listed above] | Correction: [specific correction]
(2) Error Identified: [short sentence or phrase] | Error Type: [one error type listed above] | Correction: [specific correction]
...
[End of Error Detection for Response 1]
[Error Detection for Response 2] 
If no errors found, show "No errors identified". Otherwise, list the errors one by one in the following format.
(1) Error Identified: [short sentence or phrase] | Error Type: [one error type listed above] | Correction: [specific correction]
(2) Error Identified: [short sentence or phrase] | Error Type: [one error type listed above] | Correction: [specific correction]
...
[End of Error Detection for Response 2]"""

# ========================= STAGE 2: SCORING =========================

SCORING_STAGE_PROMPT = """You are a skilled expert at scoring responses. You should evaluate given responses based on the given judging criteria.
In the previous conversation, there are two responses and the conversation context from the User as well as the error detection result from the Assistant. 
You need to refer to the [Helpfulness Scoring Guidelines] to score two response with two individual scores and a ranking score based on the [Ranking Scoring Guidelines]
Before scoring, please refer to the error detection result and analyze step by step. Your scoring needs to be as strict as possible. Please strictly follow the required output format.

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
Analysis on response 1
[The End of Analysis on Response 1]

[The Begin of Analysis on Response 2]
Analysis on response 2
[The End of Analysis on Response 2]

[The Begin of Individual Scores]
\\boxed{x, y} (the scores of each response in order)
[The End of Individual Scores]

[The Begin of Ranking Score]
\\boxed{z} 
[The End of Ranking Score]
"""

# ========================= DATA STRUCTURES =========================

@dataclass
class ErrorDetectionResult:
    """Structured error detection results for a single response."""
    error_status: str  # VERIFIED_CLEAN, MINOR_VERIFIABLE_ERRORS, MAJOR_VERIFIABLE_ERRORS, SEVERE_VERIFIABLE_ERRORS
    errors_found: str
    error_types: str
    corrections: str
    
class TwoStageMetadata(TypedDict):
    num_responses: int
    helpfulness_1: Optional[int]
    helpfulness_2: Optional[int] 
    preference_ranking: Optional[int]
    error_detection_stage_complete: Optional[bool]
    error_detection_results: Optional[str]
    context: Optional[str]
    response1: Optional[str]
    response2: Optional[str]

# ========================= PROMPT FORMATTING =========================

def format_error_detection_stage_prompt(context: str, response1: str, response2: Optional[str] = None) -> str:
    """Format the error detection stage prompt."""
    if response2 is None:
        responses = f"**Response 1:**\n{response1}"
    else:
        responses = f"**Response 1:**\n{response1}\n\n**Response 2:**\n{response2}"
    
    return ERROR_DETECTION_STAGE_PROMPT.format(
        context=context,
        responses=responses
    )

def format_scoring_stage_prompt(context: str, response1: str, response2: Optional[str], error_detection_results: str) -> str:
    """Format the scoring stage prompt with error detection results."""
    return SCORING_STAGE_PROMPT

# ========================= PARSING UTILITIES =========================

def parse_error_detection_response(response: str, num_responses: int = 2) -> Tuple[bool, str]:
    """
    Parse error detection response and extract structured content.
    Returns (success, formatted_content).
    """
    try:
        blocks = {}
        # Try to find each response block
        for idx in range(num_responses):
            response_num = idx + 1
            # Primary regex pattern - matches the exact format
            pattern = rf"\[Error Detection for Response {response_num}\]\s*(.*?)\s*\[End of Error Detection for Response {response_num}\]"
            match = re.search(pattern, response, flags=re.DOTALL | re.IGNORECASE)
            
            if match:
                content = match.group(1).strip()
                blocks[response_num] = content
        
        # If we found structured blocks, reconstruct them properly
        if blocks:
            structured_parts = []
            for response_num in sorted(blocks.keys()):
                content = blocks[response_num]
                structured_block = f"[Error Detection for Response {response_num}]\n{content}\n[End of Error Detection for Response {response_num}]"
                structured_parts.append(structured_block)
            return True, "\n\n".join(structured_parts)
        else:
            return False, "No valid error detection results found."

    except Exception as e:
        return False, f"Error parsing response: {str(e)}"


def parse_scoring_response(assistant_response: str, num_responses: int) -> tuple[bool, Optional[list], Optional[str], str]:
    """
    Validate the strict format and extract scores.
    Returns: (is_valid_format, individual_scores, preference_ranking, error_message)
    """
    if num_responses == 1:
        # Expected format: "[The Begin of Individual Scores]\n\\boxed{x} \n[The End of Individual Scores]\n"
        pattern = r'\[The Begin of Individual Scores\]\s*\n\s*\\boxed\{([^}]+)\}\s*\n\s*\[The End of Individual Scores\]'
        match = re.search(pattern, assistant_response, re.DOTALL)
        
        if not match:
            return False, [], None, "Format violation: Expected '[The Begin of Individual Scores]\\n\\\\boxed{x} \\n[The End of Individual Scores]\\n' for single response"
        
        score_content = match.group(1).strip()
        # For single response, should be just one score
        if ',' in score_content:
            return False, [], None, "Format violation: Single response should contain only one score, found comma"
        
        return True, [score_content], None, ""
        
    elif num_responses == 2:
        # Expected format: "[The Begin of Individual Scores]\n\\boxed{x, y} \n[The End of Individual Scores]\n[The Begin of Ranking Score]\n\\boxed{z} \n[The End of Ranking Score]"
        
        # Check individual scores section
        individual_pattern = r'\[The Begin of Individual Scores\]\s*\n\s*\\boxed\{([^}]+)\}\s*\n\s*\[The End of Individual Scores\]'
        individual_match = re.search(individual_pattern, assistant_response, re.DOTALL)
        
        if not individual_match:
            return False, [], None, "Format violation: Missing or incorrect '[The Begin of Individual Scores]\\n\\\\boxed{x, y} \\n[The End of Individual Scores]' section"
        
        # Check ranking score section
        ranking_pattern = r'\[The Begin of Ranking Score\]\s*\n\s*\\boxed\{([^}]+)\}\s*\n\s*\[The End of Ranking Score\]'
        ranking_match = re.search(ranking_pattern, assistant_response, re.DOTALL)
        
        if not ranking_match:
            return False, [], None, "Format violation: Missing or incorrect '[The Begin of Ranking Score]\\n\\\\boxed{z} \\n[The End of Ranking Score]' section"
        
        # Extract individual scores
        scores_content = individual_match.group(1).strip()
        individual_scores = [score.strip() for score in scores_content.split(',')]
        
        # For two responses, should have exactly two scores
        if len(individual_scores) != 2:
            return False, [], None, f"Format violation: Expected exactly 2 individual scores, found {len(individual_scores)}"
        
        # Extract preference ranking
        preference_content = ranking_match.group(1).strip()
        # Preference ranking should be a single value
        if ',' in preference_content:
            return False, [], None, "Format violation: Preference ranking should be a single value, found comma"
        
        return True, individual_scores, preference_content, ""
    
    else:
        return False, [], None, f"Unsupported number of responses: {num_responses}"



# ========================= TWO-STAGE ENVIRONMENT =========================

@ray.remote
class TwoStageErrorDetectionEnvironment(EnvironmentInterface):
    """Two-stage error detection environment: Stage 1 error detection, Stage 2 score."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.format_penalty = cfg.get("format_penalty", -20)
        self.verifiable_error_bonus_multiplier = cfg.get("verifiable_error_bonus_multiplier", 0.0)
        logging.basicConfig(level=logging.INFO)
    

    def step(self, message_log_batch: list[list[dict[str, str]]], metadata: list[TwoStageMetadata]) -> EnvironmentReturn:
        """Process two-stage error detection and scoring."""
        
        print(f"\n[TWO-STAGE ENV] Processing batch of {len(message_log_batch)} samples")
        
        rewards = []
        observations = []
        next_metadata = []
        
        for i, (conversation, meta) in enumerate(zip(message_log_batch, metadata)):
            if i < 2:  # First couple samples
                print(f"\n[SAMPLE {i}] Processing sample {i}")
                print(f"  error_detection_stage_complete: {meta.get('error_detection_stage_complete', False)}")
                print(f"  Has ground truth: h1={meta.get('helpfulness_1')}, h2={meta.get('helpfulness_2')}, pref={meta.get('preference_ranking')}")
                
            # Extract assistant's response
            rm_response = ""
            for msg in reversed(conversation):
                if msg["role"] == "assistant":
                    rm_response = msg["content"]
                    break
            if i < 2:  # First couple samples
                print(f"  Assistant response length: {len(rm_response)}")
                print(f"  Assistant response preview: {rm_response[:100]}...")
            
            # Check which stage we're in
            if not meta.get("error_detection_stage_complete"):
                # STAGE 1: Error Detection
                reward, obs, updated_meta = self._process_error_detection_stage(
                    rm_response, meta
                )
                # CRITICAL: Ensure we move to next stage
                if updated_meta and not updated_meta.get("error_detection_stage_complete"):
                    print(f"[WARNING] Sample {i} didn't complete error detection stage properly")
            else:
                # STAGE 2: Scoring
                if i < 2:  # First couple samples
                    print(f"  [STAGE 2] Processing scoring stage")
                reward, obs, updated_meta = self._process_scoring_stage(
                    rm_response, meta
                )
                if i < 2:  # First couple samples
                    print(f"  [STAGE 2] Final reward: {reward}")
            
            rewards.append(reward)
            observations.append(obs)
            next_metadata.append(updated_meta)
            if i < 2:  # First couple samples
                print(f"  Sample {i} completed with reward: {reward}")
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        # Only terminate after scoring stage
        terminateds = torch.tensor([
            meta is None for meta in next_metadata
        ], dtype=torch.bool)

        # Add stop strings for each stage
        next_stop_strings = []
        for meta in next_metadata:
            if meta and not meta.get("error_detection_stage_complete"):
                # Error detection stage - stop after error detection output
                next_stop_strings.append(["[End of Error Detection for Response 2]"])
            else:
                # Scoring stage - stop after ranking score
                next_stop_strings.append(["[The End of Ranking Score]"])
        
        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=next_stop_strings,  # Use stage-specific stop strings
            rewards=rewards_tensor,
            terminateds=terminateds,
        )

    def _process_error_detection_stage(self, response: str, metadata: TwoStageMetadata) -> Tuple[float, dict, TwoStageMetadata]:
        """Process error detection stage."""
        
        # Parse and store error detection results
        is_parsed, parsed_response = parse_error_detection_response(response)
        
        # Store error detection results and prepare for scoring stage
        updated_metadata = metadata.copy()
        updated_metadata["error_detection_stage_complete"] = True
        updated_metadata["error_detection_results"] = parsed_response
        
        # IMPORTANT: Create the observation that will be used as the next prompt
        # This should be the scoring stage prompt
        scoring_prompt = format_scoring_stage_prompt(
            metadata["context"],
            metadata["response1"], 
            metadata["response2"],
            parsed_response
        )
        
        reward = 0.0 if is_parsed else float(self.format_penalty)

        # Return observation that becomes the next user message
        obs = {"role": "user", "content": "\n<|im_start|>user\n" + scoring_prompt + "<|im_end|>\n<|im_start|>assistant\n"}
        return reward, obs, updated_metadata


    def _process_scoring_stage(self, response: str, metadata: TwoStageMetadata) -> Tuple[float, dict, TwoStageMetadata]:
        """Process scoring stage with extensive debugging."""
        # Parse scoring response (still need this for reward calculation)
        is_valid, individual_scores, preference_ranking, error_msg = parse_scoring_response(
            response, metadata["num_responses"]
        )

        if error_msg:
            print(f"  Parse error: {error_msg}")
        
        if not is_valid:
            print(f"[SCORING STAGE] Parse error: {error_msg}")
            obs = {
                "role": "environment",
                "content": f"<environment>Scoring stage format error: {error_msg}</environment>"
            }
            return float(self.format_penalty), obs, None
        
        # Calculate base reward from scoring accuracy (no fact-checking modifier)
        reward = self.format_penalty
        try:
            if metadata["num_responses"] == 1:
                if metadata["helpfulness_1"] is not None and individual_scores:
                    reward = - distance_abs(individual_scores[0], metadata["helpfulness_1"])
                    
            elif metadata["num_responses"] == 2:
                # Calculate distance for both individual scores
                if metadata["helpfulness_1"] is not None and individual_scores and len(individual_scores) == 2 and metadata["preference_ranking"] is not None and preference_ranking:
                    reward = - distance_abs(individual_scores[0], metadata["helpfulness_1"])  - distance_abs(individual_scores[1], metadata["helpfulness_2"]) - distance_abs(preference_ranking, metadata["preference_ranking"])
                
        except Exception as e:

            print(f"[SCORING STAGE] Score parsing error: {e}")
            print(f"  Scores that failed to parse: {scores}")
            obs = {
                "role": "environment",
                "content": f"<environment>Score parsing error: {e}</environment>"
            }
            return float(self.format_penalty), obs, None

        
        obs = {
            "role": "environment",
            "content": f"<environment>Two-stage completed. Final reward: {reward}</environment>",
        }
        return float(reward), obs, None  # Terminate episode




    def global_post_process_and_metrics(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]:
        """Calculate metrics for two-stage error detection approach."""
        rewards = batch.get("rewards", torch.tensor([]))
        num_samples = len(batch.get("idx", []))
        
        if len(rewards) == 0:
            return batch, {}
        
        rewards_np = rewards.numpy() if hasattr(rewards, 'numpy') else np.array(rewards)
        
        # Calculate metrics
        mean_reward = float(np.mean(rewards_np))
        format_violation_rate = float(np.mean(rewards_np == self.format_penalty))
        
        # For valid rewards, calculate performance metrics
        valid_rewards = rewards_np[rewards_np != self.format_penalty]
        if len(valid_rewards) > 0:
            mean_valid_reward = float(np.mean(valid_rewards))
            positive_reward_rate = float(np.mean(valid_rewards > 0))
            high_reward_rate = float(np.mean(valid_rewards > -5))  # Less than 5 points penalty
        else:
            mean_valid_reward = 0.0
            positive_reward_rate = 0.0
            high_reward_rate = 0.0
        
        metrics = {
            "mean_reward": mean_reward,
            "format_violation_rate": format_violation_rate,
            "mean_valid_reward": mean_valid_reward,
            "positive_reward_rate": positive_reward_rate,
            "high_reward_rate": high_reward_rate,
            "num_samples": num_samples,
            "valid_samples": len(valid_rewards),
            "approach": "two_stage_verifiable_error_detection",
        }
        
        return batch, metrics
    
    def shutdown(self):
        """Clean up resources."""
        pass