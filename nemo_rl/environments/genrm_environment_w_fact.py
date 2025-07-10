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

FACTCHECK_STAGE_PROMPT = """You are a fact-checking expert. Analyze the given responses for factual accuracy.

**Task:** Identify any factual errors and provide corrections when you know the accurate information.

**Context:** {context}

**Responses to Fact-Check:**
{responses}

**Output Format:**
**Response 1 Fact-Check:**
- Factual Status: [ACCURATE/MINOR_ISSUES/MAJOR_ERRORS/SEVERE_ERRORS]
- Issues Found: [List specific factual errors, or "None identified"]
- Corrections: [Provide correct information if you know it, or "None needed/available"]

**Response 2 Fact-Check:** (if applicable)
- Factual Status: [ACCURATE/MINOR_ISSUES/MAJOR_ERRORS/SEVERE_ERRORS] 
- Issues Found: [List specific factual errors, or "None identified"]
- Corrections: [Provide correct information if you know it, or "None needed/available"]

**Overall Assessment:**
- Most Accurate Response: [1, 2, or "Similar accuracy"]
- Key Differences: [Main factual differences between responses]"""

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

def parse_factcheck_response(response: str, num_responses: int) -> Tuple[bool, Dict[str, FactCheckResult], str]:
    """Parse fact-checking stage response."""
    try:
        results = {}
        
        # Parse Response 1
        resp1_pattern = r"\*\*Response 1 Fact-Check:\*\*\s*\n- Factual Status: ([^\n]+)\n- Issues Found: ([^\n]+)\n- Corrections: ([^\n]+)"
        match1 = re.search(resp1_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match1:
            results["response_1"] = FactCheckResult(
                factual_status=match1.group(1).strip(),
                issues_found=match1.group(2).strip(), 
                corrections=match1.group(3).strip()
            )
        else:
            return False, {}, "Could not parse Response 1 fact-check"
        
        # Parse Response 2 if needed
        if num_responses == 2:
            resp2_pattern = r"\*\*Response 2 Fact-Check:\*\*\s*\n- Factual Status: ([^\n]+)\n- Issues Found: ([^\n]+)\n- Corrections: ([^\n]+)"
            match2 = re.search(resp2_pattern, response, re.DOTALL | re.IGNORECASE)
            
            if match2:
                results["response_2"] = FactCheckResult(
                    factual_status=match2.group(1).strip(),
                    issues_found=match2.group(2).strip(),
                    corrections=match2.group(3).strip()
                )
            else:
                return False, {}, "Could not parse Response 2 fact-check"
        
        return True, results, ""
        
    except Exception as e:
        return False, {}, f"Parsing error: {str(e)}"

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
        self.factcheck_bonus_multiplier = cfg.get("factcheck_bonus_multiplier", 0.2)
        logging.basicConfig(level=logging.INFO)
    
    def step(self, message_log_batch: list[list[dict[str, str]]], metadata: list[TwoStageMetadata]) -> EnvironmentReturn:
        """Process two-stage fact-checking and scoring."""
        
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
                # STAGE 1: Fact-checking
                reward, obs, updated_meta = self._process_factcheck_stage(
                    assistant_response, meta
                )
            else:
                # STAGE 2: Scoring with fact-check results
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
        """Process fact-checking stage."""
        
        # Parse fact-checking response
        is_valid, factcheck_results, error_msg = parse_factcheck_response(
            response, metadata["num_responses"]
        )
        
        if not is_valid:
            print(f"[FACTCHECK STAGE] Parse error: {error_msg}")
            # Return penalty and stay in same stage
            return float(self.format_penalty), {
                "role": "environment",
                "content": f"Fact-check stage format error: {error_msg}"
            }, metadata
        
        # Store fact-check results and move to next stage  
        updated_metadata = metadata.copy()
        updated_metadata["factcheck_stage_complete"] = True
        updated_metadata["factcheck_results"] = json.dumps({
            key: {
                "factual_status": result.factual_status,
                "issues_found": result.issues_found, 
                "corrections": result.corrections
            } for key, result in factcheck_results.items()
        })
        
        print(f"[FACTCHECK STAGE] Completed successfully")
        print(f"  Results: {list(factcheck_results.keys())}")
        for key, result in factcheck_results.items():
            print(f"  {key}: {result.factual_status}")
        
        # No reward yet - waiting for scoring stage
        return 0.0, {
            "role": "environment", 
            "content": "Fact-checking stage completed. Proceeding to scoring stage."
        }, updated_metadata
    
    def _process_scoring_stage(self, response: str, metadata: TwoStageMetadata) -> Tuple[float, dict, TwoStageMetadata]:
        """Process scoring stage with fact-check results."""
        
        # Parse scoring response
        is_valid, scores, ranking, error_msg = parse_scoring_response(
            response, metadata["num_responses"]
        )
        
        if not is_valid:
            print(f"[SCORING STAGE] Parse error: {error_msg}")
            return float(self.format_penalty), {
                "role": "environment",
                "content": f"Scoring stage format error: {error_msg}"
            }, None  # Terminate episode
        
        # Calculate base reward from scoring accuracy
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
        
        # Apply fact-checking bonus/penalty
        factcheck_modifier = 1.0
        if metadata.get("factcheck_results"):
            try:
                factcheck_data = json.loads(metadata["factcheck_results"])
                factcheck_results_parsed = {
                    key: FactCheckResult(**data) for key, data in factcheck_data.items()
                }
                factcheck_modifier = calculate_factcheck_modifier(factcheck_results_parsed)
                
            except Exception as e:
                print(f"[SCORING STAGE] Factcheck modifier error: {e}")
        
        # Final reward combines base accuracy with fact-checking quality
        factcheck_bonus = base_reward * factcheck_modifier * self.factcheck_bonus_multiplier
        final_reward = base_reward + factcheck_bonus
        
        print(f"[SCORING STAGE] Completed")
        print(f"  Base reward: {base_reward}")
        print(f"  Factcheck modifier: {factcheck_modifier:.2f}")
        print(f"  Factcheck bonus: {factcheck_bonus:.2f}")
        print(f"  Final reward: {final_reward}")
        
        return float(final_reward), {
            "role": "environment",
            "content": f"Two-stage evaluation complete. Base: {base_reward}, Factcheck bonus: {factcheck_bonus:.1f}, Final: {final_reward}"
        }, None  # Terminate episode
    
    def global_post_process_and_metrics(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]:
        """Calculate metrics for two-stage approach."""
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
            high_reward_rate = float(np.mean(valid_rewards > 5))  # Arbitrary threshold
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
            "factcheck_bonus_multiplier": self.factcheck_bonus_multiplier,
        }
        
        return batch, metrics
    
    def shutdown(self):
        """Clean up resources."""
        pass

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

# ========================= EXAMPLE USAGE =========================

def example_two_stage_usage():
    """Example of how to use the two-stage system."""
    
    # Stage 1: Generate fact-check
    context = "What is the capital of France?"
    response1 = "The capital of France is Paris, which has been the capital since 1789."
    response2 = "The capital of France is Lyon, a major city in the south."
    
    factcheck_prompt = format_factcheck_stage_prompt(context, response1, response2)
    print("=== STAGE 1: FACT-CHECKING ===")
    print(factcheck_prompt)
    
    # Mock fact-check response (in practice, this comes from model generation)
    mock_factcheck_response = """**Response 1 Fact-Check:**
- Factual Status: MINOR_ISSUES
- Issues Found: Paris became capital much earlier than 1789, around 987 AD
- Corrections: Paris has been the de facto capital since around 987 AD

**Response 2 Fact-Check:**
- Factual Status: MAJOR_ERRORS
- Issues Found: Lyon is not the capital of France, Paris is the capital
- Corrections: The capital of France is Paris, not Lyon

**Overall Assessment:**
- Most Accurate Response: 1
- Key Differences: Response 1 is mostly correct with wrong date, Response 2 has fundamental error"""
    
    # Parse fact-check results
    is_valid, results, error = parse_factcheck_response(mock_factcheck_response, 2)
    print(f"\nFact-check parsing successful: {is_valid}")
    if is_valid:
        for key, result in results.items():
            print(f"{key}: {result.factual_status}")
    
    # Stage 2: Generate scores using fact-check results
    scoring_prompt = format_scoring_stage_prompt(context, response1, response2, mock_factcheck_response)
    print("\n=== STAGE 2: SCORING ===")
    print(scoring_prompt)
    
    return factcheck_prompt, scoring_prompt

if __name__ == "__main__":
    example_two_stage_usage()