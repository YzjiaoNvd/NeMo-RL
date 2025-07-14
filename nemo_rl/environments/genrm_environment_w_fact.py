# Two-Stage Fact-Checking GenRM Implementation
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

# ========================= STAGE 1: FACT-CHECKING =========================

FACTCHECK_STAGE_PROMPT = """You are a fact-checking expert. Analyze the given responses for factual accuracy. Keep your reply brief and strictly follow the required output format. 

**Task:** Identify any factual errors and provide corrections when you know the accurate information.

**Context:** {context}

**Responses to Fact-Check:**
{responses}

**Output Format:**
**Response 1**
- Factual Errors Found: [List specific factual errors, or "None identified"]
- Corrections: [Provide correct information if you know it, or "Unknown"]

**Response 2** (if applicable)
- Factual Errors Found: [List specific factual errors, or "None identified"]
- Corrections: [Provide correct information if you know it, or "Unknown"]"""

# ========================= STAGE 2: SCORING =========================

SCORING_STAGE_PROMPT = """You are a response quality evaluator. Score these responses considering both helpfulness and factual accuracy.

**Scoring Guidelines (1-5):**
- **5**: Extremely helpful and factually accurate
- **4**: Very helpful with minor factual imprecisions  
- **3**: Moderately helpful but some factual concerns
- **2**: Limited helpfulness due to factual issues
- **1**: Not helpful or contains serious factual errors

**Ranking Guidelines (1-6 for two responses):**
1 = Response 1 much better | 2 = Response 1 better | 3 = Response 1 slightly better
4 = Response 2 slightly better | 5 = Response 2 better | 6 = Response 2 much better

**Context:** {context}

**Responses:**
{responses}

**Fact-Check Results:**
{factcheck_results}

**Analysis and Scoring:**

[Analysis Response 1]
[Consider both helpfulness and factual accuracy from fact-check]
[End Analysis Response 1]

[Analysis Response 2] (if applicable)
[Consider both helpfulness and factual accuracy from fact-check]  
[End Analysis Response 2]

[Individual Scores]
\\boxed{{{scores_format}}}
[End Individual Scores]

[Ranking Score] (if two responses)
\\boxed{{ranking}}
[End Ranking Score]"""

# ========================= DATA STRUCTURES =========================

@dataclass
class FactCheckResult:
    """Structured fact-check results for a single response."""
    factual_status: str  # ACCURATE, MINOR_ISSUES, MAJOR_ERRORS, SEVERE_ERRORS
    issues_found: str
    corrections: str
    
class TwoStageMetadata(TypedDict):
    num_responses: int
    helpfulness_1: Optional[int]
    helpfulness_2: Optional[int] 
    preference_ranking: Optional[int]
    factcheck_stage_complete: Optional[bool]
    factcheck_results: Optional[str]

# ========================= PROMPT FORMATTING =========================

def format_factcheck_stage_prompt(context: str, response1: str, response2: Optional[str] = None) -> str:
    """Format the fact-checking stage prompt."""
    if response2 is None:
        responses = f"**Response 1:**\n{response1}"
    else:
        responses = f"**Response 1:**\n{response1}\n\n**Response 2:**\n{response2}"
    
    return FACTCHECK_STAGE_PROMPT.format(
        context=context,
        responses=responses
    )

def format_scoring_stage_prompt(context: str, response1: str, response2: Optional[str], factcheck_results: str) -> str:
    """Format the scoring stage prompt with fact-check results."""
    if response2 is None:
        responses = f"**Response 1:**\n{response1}"
        scores_format = "x"
    else:
        responses = f"**Response 1:**\n{response1}\n\n**Response 2:**\n{response2}"
        scores_format = "x, y"
    
    return SCORING_STAGE_PROMPT.format(
        context=context,
        responses=responses,
        factcheck_results=factcheck_results,
        scores_format=scores_format
    )

# ========================= PARSING UTILITIES =========================

def parse_scoring_response(response: str, num_responses: int) -> Tuple[bool, list, Optional[str], str]:
    """Parse scoring stage response."""
    try:
        # Extract individual scores
        scores_pattern = r"\[Individual Scores\]\s*\\boxed\{([^}]+)\}\s*\[End Individual Scores\]"
        scores_match = re.search(scores_pattern, response, re.DOTALL)
        
        if not scores_match:
            return False, [], None, "Could not find individual scores"
        
        scores_str = scores_match.group(1).strip()
        scores = [s.strip() for s in scores_str.split(',')]
        
        # Validate score count
        if num_responses == 1 and len(scores) != 1:
            return False, [], None, f"Expected 1 score, got {len(scores)}"
        elif num_responses == 2 and len(scores) != 2:
            return False, [], None, f"Expected 2 scores, got {len(scores)}"
        
        # Extract ranking if two responses
        ranking = None
        if num_responses == 2:
            ranking_pattern = r"\[Ranking Score\]\s*\\boxed\{([^}]+)\}\s*\[End Ranking Score\]"
            ranking_match = re.search(ranking_pattern, response, re.DOTALL)
            
            if ranking_match:
                ranking = ranking_match.group(1).strip()
            else:
                return False, [], None, "Missing ranking score for two responses"
        
        return True, scores, ranking, ""
        
    except Exception as e:
        return False, [], None, f"Scoring parse error: {str(e)}"

# ========================= FACT-CHECK SCORING =========================

def calculate_factcheck_modifier(factcheck_results: Dict[str, FactCheckResult]) -> float:
    """Calculate modifier based on fact-checking results."""
    status_scores = {
        "ACCURATE": 1.0,
        "MINOR_ISSUES": 0.8,
        "MAJOR_ERRORS": 0.5,
        "SEVERE_ERRORS": 0.2
    }
    
    total_modifier = 0.0
    count = 0
    
    for result in factcheck_results.values():
        status = result.factual_status.upper()
        modifier = status_scores.get(status, 0.5)  # Default to moderate penalty
        total_modifier += modifier
        count += 1
    
    return total_modifier / count if count > 0 else 1.0

# ========================= TWO-STAGE ENVIRONMENT =========================

@ray.remote
class TwoStageFactCheckEnvironment(EnvironmentInterface):
    """Two-stage fact-checking environment: Stage 1 fact-check, Stage 2 score."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.format_penalty = cfg.get("format_penalty", -100)
        self.factcheck_bonus_multiplier = cfg.get("factcheck_bonus_multiplier", 0.0)
        logging.basicConfig(level=logging.INFO)
    

    def step(self, message_log_batch: list[list[dict[str, str]]], metadata: list[TwoStageMetadata]) -> EnvironmentReturn:
        """Process two-stage fact-checking and scoring - simplified version."""
        
        rewards = []
        observations = []
        next_metadata = []
        
        for conversation, meta in zip(message_log_batch, metadata):
            # Extract assistant's response
            assistant_response = ""
            for msg in conversation:
                if msg["role"] == "assistant":
                    assistant_response = msg["content"]
                    break
            
            # Check which stage we're in
            if not meta.get("factcheck_stage_complete", False):
                # STAGE 1: Fact-checking (no parsing, just store raw response)
                reward, obs, updated_meta = self._process_factcheck_stage(
                    assistant_response, meta
                )
            else:
                # STAGE 2: Scoring with raw fact-check results
                reward, obs, updated_meta = self._process_scoring_stage(
                    assistant_response, meta
                )
            
            rewards.append(reward)
            observations.append(obs)
            next_metadata.append(updated_meta)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        # Only terminate after scoring stage
        terminateds = torch.tensor([
            meta.get("factcheck_stage_complete", False) for meta in next_metadata
        ], dtype=torch.bool)
        
        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards_tensor,
            terminateds=terminateds,
        )


    def _process_factcheck_stage(self, response: str, metadata: TwoStageMetadata) -> Tuple[float, dict, TwoStageMetadata]:
        """Process fact-checking stage - simplified version without parsing."""
        
        # Debug: Show response length and first part
        print(f"[FACTCHECK STAGE] Raw response length: {len(response)}")
        if len(response) > 0:
            print(f"[FACTCHECK STAGE] First 200 chars: {response[:200]}...")
        
        # Truncate if too long (keep first 2000 chars)
        max_factcheck_length = 2000
        truncated_response = response
        if len(response) > max_factcheck_length:
            truncated_response = response[:max_factcheck_length] + "\n[...truncated due to length]"
            print(f"[FACTCHECK STAGE] Truncated response from {len(response)} to {len(truncated_response)} chars")
        
        # Store raw fact-check results and move to scoring stage  
        updated_metadata = metadata.copy()
        updated_metadata["factcheck_stage_complete"] = True
        updated_metadata["factcheck_results"] = truncated_response  # Store raw response directly
        
        print(f"[FACTCHECK STAGE] Completed successfully - stored raw response")
        
        # No reward yet - waiting for scoring stage
        return 0.0, {
            "role": "environment", 
            "content": "Fact-checking stage completed. Proceeding to scoring stage."
        }, updated_metadata



    def _process_scoring_stage(self, response: str, metadata: TwoStageMetadata) -> Tuple[float, dict, TwoStageMetadata]:
        """Process scoring stage with raw fact-check results - simplified version."""
        
        # Parse scoring response (still need this for reward calculation)
        is_valid, scores, ranking, error_msg = parse_scoring_response(
            response, metadata["num_responses"]
        )
        
        if not is_valid:
            print(f"[SCORING STAGE] Parse error: {error_msg}")
            return float(self.format_penalty), {
                "role": "environment",
                "content": f"Scoring stage format error: {error_msg}"
            }, None  # Terminate episode
        
        # Calculate base reward from scoring accuracy (no fact-checking modifier)
        base_reward = 0.0
        try:
            if metadata["num_responses"] == 1:
                if metadata["helpfulness_1"] is not None:
                    base_reward = -abs(int(scores[0]) - metadata["helpfulness_1"])
            
            elif metadata["num_responses"] == 2:
                if metadata["helpfulness_1"] is not None and metadata["helpfulness_2"] is not None:
                    base_reward = (
                        -abs(int(scores[0]) - metadata["helpfulness_1"]) 
                        -abs(int(scores[1]) - metadata["helpfulness_2"])
                    )
                
                if metadata["preference_ranking"] is not None and ranking:
                    base_reward -= abs(int(ranking) - metadata["preference_ranking"])
                        
        except ValueError as e:
            print(f"[SCORING STAGE] Score parsing error: {e}")
            return float(self.format_penalty), {
                "role": "environment",
                "content": f"Score parsing error: {e}"
            }, None
        
        # Final reward is just the base accuracy (no fact-checking bonus/penalty)
        final_reward = base_reward
        
        # Get raw fact-checking results for logging
        raw_factcheck = metadata.get("factcheck_results", "")
        factcheck_length = len(raw_factcheck)
        
        print(f"[SCORING STAGE] Completed")
        print(f"  Base reward: {base_reward}")
        print(f"  Final reward: {final_reward}")
        print(f"  Used fact-check input length: {factcheck_length} chars")
        
        return float(final_reward), {
            "role": "environment",
            "content": f"Two-stage evaluation complete. Final reward: {final_reward} (used {factcheck_length} char fact-check input)"
        }, None  # Terminate episode




    def global_post_process_and_metrics(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]:
        """Calculate metrics for simplified two-stage approach."""
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
            "approach": "simplified_two_stage",  # Indicate this is the simplified version
        }
        
        return batch, metrics
    


# ========================= INTEGRATION UTILITIES =========================

def create_factcheck_stage_data(context: str, response1: str, response2: Optional[str] = None) -> dict:
    """Create data for fact-checking stage."""
    prompt = format_factcheck_stage_prompt(context, response1, response2)
    
    return {
        "prompt": prompt,
        "num_responses": 2 if response2 else 1,
        "stage": "factcheck"
    }

def create_scoring_stage_data(context: str, response1: str, response2: Optional[str], factcheck_results: str) -> dict:
    """Create data for scoring stage."""
    prompt = format_scoring_stage_prompt(context, response1, response2, factcheck_results)
    
    return {
        "prompt": prompt, 
        "num_responses": 2 if response2 else 1,
        "stage": "scoring",
        "factcheck_results": factcheck_results
    }

