# Three-Stage GenRM Implementation: Vanilla → Fact-Check → Enhanced
# Modified to support loading stage 1 results and starting from stage 2
import re
import json
import logging
from typing import Any, Optional, TypedDict, Tuple, Dict, List
import numpy as np
import ray
import torch
from dataclasses import dataclass

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)

# ========================= STAGE 1: VANILLA GENRM =========================

VANILLA_GENRM_PROMPT = """You are a skilled expert at scoring responses. You should evaluate given responses based on the given judging criteria.
Given the context of the conversation (the last turn is the User's query) and two responses from the Assistant, you need to refer to the [Helpfulness Scoring Guidelines] to score each individual response.
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
6 = Response 2 is much better than Response 2

#### Conversation Context ####
{context}

#### Responses to be Scored ####
{responses}

#### Output Format Requirements ####
[The Begin of Analysis on Response 1]
Analysis on response 1
[The End of Analysis on Response 1]

[The Begin of Analysis on Response 2]
Analysis on response 2
[The End of Analysis on Response 2]

[The Begin of Individual Scores]
\\boxed{{x, y}}  (the scores of each response in order)
[The End of Individual Scores]

[The Begin of Ranking Score]
\\boxed{{z}} 
[The End of Ranking Score]
"""

# ========================= STAGE 2: FACT-CHECKING =========================

FACTCHECK_PROMPT = """You are a fact-checking expert. Analyze the given two responses for factual accuracy. Strictly follow the required output format and make your answer as brief as possible. 

**Task:** Identify any factual errors and provide corrections when you know the accurate information.

**Context:** 
{context}

**Responses:**
{responses}

**Output Format:**
[Fact Checking for Response 1]
- Factual Errors: [List all factual errors (the spans from the original model responses), or "None identified"]
- Corrections: [Provide correct information if you know it, or "Unknown"]
[End of Fact Checking for Response 1]

[Fact Checking for Response 2] 
- Factual Errors: [List all factual errors (the spans from the original model responses), or "None identified"]
- Corrections: [Provide correct information if you know it, or "Unknown"]
[End of Fact Checking for Response 2]"""

# ========================= STAGE 3: ENHANCED GENRM =========================

ENHANCED_GENRM_PROMPT = """You are a skilled expert at scoring responses. You should evaluate given responses based on the given judging criteria.
Given the context of the conversation (the last turn is the User's query), two responses from the Assistant, and the fact checking results for these responses, you need to refer to the [Helpfulness Scoring Guidelines] to score each individual response.
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

#### Fact Checking Results as Reference ####
{factcheck_results}

#### Output Format Requirements ####
[The Begin of Analysis on Response 1]
Analysis on response 1
[The End of Analysis on Response 1]

[The Begin of Analysis on Response 2]
Analysis on response 2
[The End of Analysis on Response 2]

[The Begin of Individual Scores]
\\boxed{{x, y}}  (the scores of each response in order)
[The End of Individual Scores]

[The Begin of Ranking Score]
\\boxed{{z}} 
[The End of Ranking Score]
"""

# ========================= DATA STRUCTURES =========================

class ThreeStageMetadata(TypedDict):
    num_responses: int
    helpfulness_1: Optional[int]
    helpfulness_2: Optional[int] 
    preference_ranking: Optional[int]
    stage: str  # "vanilla", "factcheck", "enhanced"
    vanilla_scores: Optional[List[int]]
    vanilla_ranking: Optional[int]
    vanilla_response: Optional[str]  # Store original vanilla response
    factcheck_results: Optional[str]
    context: Optional[str]
    response1: Optional[str]
    response2: Optional[str]

# ========================= PROMPT FORMATTING =========================

def format_vanilla_genrm_prompt(context: str, response1: str, response2: Optional[str] = None) -> str:
    """Format the vanilla GenRM prompt."""
    if response2 is None:
        responses = f"**Response 1:**\n{response1}"
    else:
        responses = f"**Response 1:**\n{response1}\n\n**Response 2:**\n{response2}"
    
    return VANILLA_GENRM_PROMPT.format(
        context=context,
        responses=responses
    )

def format_factcheck_prompt(context: str, response1: str, response2: Optional[str] = None) -> str:
    """Format the fact-checking prompt."""
    if response2 is None:
        responses = f"**Response 1:**\n{response1}"
    else:
        responses = f"**Response 1:**\n{response1}\n\n**Response 2:**\n{response2}"
    
    return FACTCHECK_PROMPT.format(
        context=context,
        responses=responses
    )

def format_enhanced_genrm_prompt(context: str, response1: str, response2: Optional[str], factcheck_results: str) -> str:
    """Format the enhanced GenRM prompt with fact-check results."""
    if response2 is None:
        responses = f"**Response 1:**\n{response1}"
    else:
        responses = f"**Response 1:**\n{response1}\n\n**Response 2:**\n{response2}"
    
    return ENHANCED_GENRM_PROMPT.format(
        context=context,
        responses=responses,
        factcheck_results=factcheck_results
    )

# ========================= PARSING UTILITIES =========================

def parse_genrm_response(response: str, num_responses: int) -> Tuple[bool, List[str], Optional[str], str]:
    """Parse GenRM response and extract scores and ranking."""
    try:
        # Extract individual scores
        scores_pattern = r"\[The Begin of Individual Scores\]\s*\\boxed\{([^}]+)\}\s*.*?\[The End of Individual Scores\]"
        scores_match = re.search(scores_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if not scores_match:
            return False, [], None, "Could not find individual scores section"
        
        scores_str = scores_match.group(1).strip()
        scores = [s.strip() for s in scores_str.split(',')]
        
        # Validate score count
        if num_responses == 1 and len(scores) != 1:
            return False, [], None, f"Expected 1 score for single response, got {len(scores)}"
        elif num_responses == 2 and len(scores) != 2:
            return False, [], None, f"Expected 2 scores for two responses, got {len(scores)}"
        
        # Extract ranking if two responses
        ranking = None
        if num_responses == 2:
            ranking_pattern = r"\[The Begin of Ranking Score\]\s*\\boxed\{([^}]+)\}\s*.*?\[The End of Ranking Score\]"
            ranking_match = re.search(ranking_pattern, response, re.DOTALL | re.IGNORECASE)
            
            if not ranking_match:
                return False, scores, None, "Missing ranking score section"
            
            ranking = ranking_match.group(1).strip()
            
            if ',' in ranking:
                return False, scores, None, "Ranking should be a single value"
        
        return True, scores, ranking, ""
        
    except Exception as e:
        return False, [], None, f"Parse error: {str(e)}"

def parse_factcheck_response(response: str, num_responses: int = 2) -> Tuple[bool, str]:
    """Parse fact-checking response and extract structured content."""
    try:
        blocks = {}
        for idx in range(num_responses):
            response_num = idx + 1
            pattern = rf"\[Fact Checking for Response {response_num}\]\s*(.*?)\s*\[End of Fact Checking for Response {response_num}\]"
            match = re.search(pattern, response, flags=re.DOTALL | re.IGNORECASE)
            
            if match:
                content = match.group(1).strip()
                blocks[response_num] = content
        
        if blocks:
            structured_parts = []
            for response_num in sorted(blocks.keys()):
                content = blocks[response_num]
                structured_block = f"[Fact Checking for Response {response_num}]\n{content}\n[End of Fact Checking for Response {response_num}]"
                structured_parts.append(structured_block)
            return True, "\n\n".join(structured_parts)
        else:
            return False, "No valid fact checking results found."
            
    except Exception as e:
        return False, f"Fact-check parse error: {str(e)}"

def calculate_reward_from_scores(scores: List[str], ranking: Optional[str], metadata: ThreeStageMetadata) -> float:
    """Calculate reward based on score accuracy."""
    total_distance = 0
    
    try:
        if metadata["num_responses"] == 1:
            if metadata["helpfulness_1"] is not None and scores and len(scores) > 0:
                score_1 = int(scores[0])
                distance_1 = abs(score_1 - metadata["helpfulness_1"])
                total_distance += distance_1
        
        elif metadata["num_responses"] == 2:
            if metadata["helpfulness_1"] is not None and metadata["helpfulness_2"] is not None and scores and len(scores) >= 2:
                score_1 = int(scores[0])
                score_2 = int(scores[1])
                distance_1 = abs(score_1 - metadata["helpfulness_1"])
                distance_2 = abs(score_2 - metadata["helpfulness_2"])
                total_distance += distance_1 + distance_2
            
            if metadata["preference_ranking"] is not None and ranking:
                ranking_int = int(ranking)
                ranking_distance = abs(ranking_int - metadata["preference_ranking"])
                total_distance += ranking_distance
        
        return -total_distance
        
    except ValueError as e:
        logging.error(f"Error calculating reward: {e}")
        return -100  # Format penalty

# ========================= THREE-STAGE ENVIRONMENT =========================

@ray.remote
class ThreeStageGenRMEnvironment(EnvironmentInterface):
    """Three-stage environment: Vanilla GenRM → Fact-check → Enhanced GenRM."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.format_penalty = cfg.get("format_penalty", -100)
        logging.basicConfig(level=logging.INFO)
        logging.info("Initialized Three-Stage GenRM Environment")
    
    def step(self, message_log_batch: list[list[dict[str, str]]], metadata: list[ThreeStageMetadata]) -> EnvironmentReturn:
        """Process three-stage evaluation."""
        
        rewards = []
        observations = []
        next_metadata = []
        
        for i, (conversation, meta) in enumerate(zip(message_log_batch, metadata)):
            # Extract assistant's response
            assistant_response = ""
            for msg in reversed(conversation):
                if msg["role"] == "assistant":
                    assistant_response = msg["content"]
                    break
            
            current_stage = meta.get("stage", "vanilla")
            
            if current_stage == "vanilla":
                # STAGE 1: Process vanilla GenRM response
                reward, obs, updated_meta = self._process_vanilla_stage(assistant_response, meta)
            elif current_stage == "factcheck":
                # STAGE 2: Process fact-checking response
                reward, obs, updated_meta = self._process_factcheck_stage(assistant_response, meta)
            elif current_stage == "enhanced":
                # STAGE 3: Process enhanced GenRM response
                reward, obs, updated_meta = self._process_enhanced_stage(assistant_response, meta)
            else:
                # Unknown stage
                reward = self.format_penalty
                obs = {"role": "environment", "content": f"Unknown stage: {current_stage}"}
                updated_meta = None
            
            rewards.append(reward)
            observations.append(obs)
            next_metadata.append(updated_meta)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        # Only terminate after enhanced stage
        terminateds = torch.tensor([
            meta is None or meta.get("stage") == "complete" for meta in next_metadata
        ], dtype=torch.bool)
        
        # Stage-specific stop strings
        next_stop_strings = []
        for meta in next_metadata:
            if meta:
                stage = meta.get("stage", "vanilla")
                if stage == "factcheck":
                    next_stop_strings.append(["[End of Fact Checking for Response 2]"])
                elif stage == "enhanced":
                    next_stop_strings.append(["[The End of Ranking Score]"])
                else:
                    next_stop_strings.append(["[The End of Ranking Score]"])
            else:
                next_stop_strings.append(None)
        
        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminateds,
        )
    
    def _process_vanilla_stage(self, response: str, metadata: ThreeStageMetadata) -> Tuple[float, dict, ThreeStageMetadata]:
        """Process Stage 1: Vanilla GenRM."""
        
        # Parse vanilla GenRM response
        is_valid, scores, ranking, error_msg = parse_genrm_response(response, metadata["num_responses"])
        
        if not is_valid:
            return float(self.format_penalty), {
                "role": "environment",
                "content": f"Vanilla GenRM format error: {error_msg}"
            }, None
        
        # Store vanilla results
        updated_metadata = metadata.copy()
        updated_metadata["stage"] = "factcheck"
        updated_metadata["vanilla_scores"] = [int(s) for s in scores] if scores else []
        updated_metadata["vanilla_ranking"] = int(ranking) if ranking else None
        updated_metadata["vanilla_response"] = response
        
        # Create fact-checking prompt for next stage
        factcheck_prompt = format_factcheck_prompt(
            metadata["context"],
            metadata["response1"],
            metadata["response2"]
        )
        
        # No reward yet - just transition to next stage
        return 0.0, {
            "role": "environment",
            "content": factcheck_prompt
        }, updated_metadata
    
    def _process_factcheck_stage(self, response: str, metadata: ThreeStageMetadata) -> Tuple[float, dict, ThreeStageMetadata]:
        """Process Stage 2: Fact-checking."""
        
        # Parse fact-checking response
        is_valid, factcheck_results = parse_factcheck_response(response, metadata["num_responses"])
        
        # Store fact-check results and prepare for enhanced stage
        updated_metadata = metadata.copy()
        updated_metadata["stage"] = "enhanced"
        updated_metadata["factcheck_results"] = factcheck_results
        
        # Create enhanced GenRM prompt for final stage
        enhanced_prompt = format_enhanced_genrm_prompt(
            metadata["context"],
            metadata["response1"],
            metadata["response2"],
            factcheck_results
        )
        
        # No reward yet - just transition to next stage
        return 0.0, {
            "role": "environment",
            "content": enhanced_prompt
        }, updated_metadata
    
    def _process_enhanced_stage(self, response: str, metadata: ThreeStageMetadata) -> Tuple[float, dict, ThreeStageMetadata]:
        """Process Stage 3: Enhanced GenRM and calculate final reward."""
        
        # Parse enhanced GenRM response
        is_valid, enhanced_scores, enhanced_ranking, error_msg = parse_genrm_response(response, metadata["num_responses"])
        
        if not is_valid:
            return float(self.format_penalty), {
                "role": "environment",
                "content": f"Enhanced GenRM format error: {error_msg}"
            }, None
        
        # Calculate rewards using vanilla scores (either pre-computed or from stage 1)
        try:
            # Get vanilla scores (either from processing or pre-loaded)
            vanilla_scores = metadata.get("vanilla_scores", [])
            vanilla_ranking = metadata.get("vanilla_ranking")
            
            if not vanilla_scores:
                logging.error("No vanilla scores found in metadata")
                return float(self.format_penalty), {
                    "role": "environment",
                    "content": "Missing vanilla scores from stage 1"
                }, None
            
            # Calculate initial reward from vanilla scores
            vanilla_scores_str = [str(s) for s in vanilla_scores]
            vanilla_ranking_str = str(vanilla_ranking) if vanilla_ranking is not None else None
            initial_reward = calculate_reward_from_scores(vanilla_scores_str, vanilla_ranking_str, metadata)
            
            # Calculate enhanced reward from enhanced scores
            enhanced_reward = calculate_reward_from_scores(enhanced_scores, enhanced_ranking, metadata)
            
            # Calculate fact-check bonus: improvement from initial to enhanced
            fact_check_bonus = enhanced_reward - initial_reward
            
            # Final reward formula: Base_reward + Fact_check_bonus
            # Where Base_reward = enhanced_reward (the final accuracy)
            base_reward = enhanced_reward
            final_reward = base_reward + fact_check_bonus
            
            # This simplifies to: final_reward = 2 * enhanced_reward - initial_reward
            
            reward_breakdown = {
                "initial_reward": initial_reward,
                "enhanced_reward": enhanced_reward,
                "fact_check_bonus": fact_check_bonus,
                "base_reward": base_reward,
                "final_reward": final_reward,
                "vanilla_scores": vanilla_scores,
                "vanilla_ranking": vanilla_ranking,
                "enhanced_scores": [int(s) for s in enhanced_scores] if enhanced_scores else [],
                "enhanced_ranking": int(enhanced_ranking) if enhanced_ranking else None,
                "is_validation": metadata.get("vanilla_response") is None,  # Indicate if this was validation
            }
            
        except Exception as e:
            logging.error(f"Error calculating three-stage reward: {e}")
            return float(self.format_penalty), {
                "role": "environment",
                "content": f"Reward calculation error: {e}"
            }, None
        
        return float(final_reward), {
            "role": "environment",
            "content": f"Three-stage evaluation complete. {reward_breakdown}"
        }, None  # Terminate episode
    
    def global_post_process_and_metrics(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]:
        """Calculate metrics for three-stage approach."""
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
            positive_bonus_rate = float(np.mean(valid_rewards > 0))  # Fact-checking helped
            high_reward_rate = float(np.mean(valid_rewards > -5))
        else:
            mean_valid_reward = 0.0
            positive_bonus_rate = 0.0
            high_reward_rate = 0.0
        
        metrics = {
            "mean_reward": mean_reward,
            "format_violation_rate": format_violation_rate,
            "mean_valid_reward": mean_valid_reward,
            "positive_bonus_rate": positive_bonus_rate,  # Rate of fact-checking improvement
            "high_reward_rate": high_reward_rate,
            "num_samples": num_samples,
            "valid_samples": len(valid_rewards),
            "approach": "three_stage_genrm_from_stage1",  # Indicate this starts from stage 1 results
        }
        
        return batch, metrics
    
    def shutdown(self):
        """Clean up resources."""
        pass