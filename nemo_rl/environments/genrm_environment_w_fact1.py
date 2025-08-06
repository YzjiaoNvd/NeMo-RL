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
FACTCHECK_STAGE_PROMPT = """You are a fact-checking expert. Your task is to evaluate the factual accuracy of two responses by identifying and verifying specific factual claims.

**Instructions:**
1. Extract ONLY verifiable factual claims (dates, numbers, names, locations, specific events, scientific facts)
2. Do NOT include opinions, subjective statements, or general knowledge
3. Categorize each claim using these definitions:
   - **Correct**: The claim is factually accurate
   - **Wrong**: The claim contains factual errors (provide the correct information)
   - **Unknown**: Cannot be verified with available knowledge

**Context:** 
{context}

**Responses to Analyze:**
{responses}

**Required Output Format:**

[Fact Checking for Response 1]
(1) Factual Claim: [exact text from response] | Status: [Correct/Wrong/Unknown] | Correction: [provide accurate information if wrong]
[Continue for all factual claims found]
[End of Fact Checking for Response 1]

[Fact Checking for Response 2]
(1) Factual Claim: [exact text from response] | Status: [Correct/Wrong/Unknown] | Correction: [provide accurate information if wrong]
[Continue for all factual claims found]
[End of Fact Checking for Response 2]

**Example:**
[Fact Checking for Response 1]
(1) Factual Claim: 1990 | Status: Correct
(2) Factual Claim: the capital of China is Shanghai | Status: Wrong | Correction: the capital of China is Beijing
[End of Fact Checking for Response 1]

[Fact Checking for Response 2]
(1) Factual Claim: the capital of China is Beijing | Status: Correct
[End of Fact Checking for Response 2]
"""
  

  

# ========================= STAGE 2: SCORING =========================

SCORING_STAGE_PROMPT = """You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.
In the previous conversation, there are two responses and the conversation context from the User as well as and the fact checking result from the Assistant. 
You need to refer to the [Helpfulness Scoring Guidelines] to score two response with two individual scores and a ranking score based on the [Ranking Scoring Guidelines]
Before scoring, please refer to the fact checking result and analyze step by step. Your scoring needs to be as strict as possible. Please strictly follow the required output format.

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
    context: Optional[str]
    response1: Optional[str]
    response2: Optional[str]

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
    '''
    if response2 is None:
        responses = f"**Response 1:**\n{response1}"
        scores_format = "x"
    else:
        responses = f"**Response 1:**\n{response1}\n\n**Response 2:**\n{response2}"
        scores_format = "x, y"
    
    return SCORING_STAGE_PROMPT.format(
        context=context,
        responses=responses,
        scores_format=scores_format
    )
    '''
    return SCORING_STAGE_PROMPT

# ========================= PARSING UTILITIES =========================

def parse_fact_checking_response(response: str, num_responses: int = 2) -> str:
    """
    Parse fact-checking response and extract structured content.
    Returns the formatted fact-checking blocks, or the raw response if parsing fails.
    """
    try:
        blocks = {}
        # Try to find each response block
        for idx in range(num_responses):
            response_num = idx + 1
            # Primary regex pattern - matches the exact format
            pattern = rf"\[Fact Checking for Response {response_num}\]\s*(.*?)\s*\[End of Fact Checking for Response {response_num}\]"
            match = re.search(pattern, response, flags=re.DOTALL | re.IGNORECASE)
            
            if match:
                content = match.group(1).strip()
                blocks[response_num] = content
        
        # If we found structured blocks, reconstruct them properly
        structured_parts = []
        for response_num in blocks:
            content = blocks[response_num]
            structured_block = f"[Fact Checking for Response {response_num}]\n{content}\n[End of Fact Checking for Response {response_num}]"
            structured_parts.append(structured_block)
        return True, "\n\n".join(structured_parts)

    except Exception as e:
        return False, "No valid fact checking results."

def parse_scoring_response(response: str, num_responses: int) -> Tuple[bool, list, Optional[str], str]:
    try:
        # Extract individual scores using updated format
        scores_pattern = r"\[The Begin of Individual Scores\]\s*\\boxed\{(\d+,\s*\d+)\}\s*\[The End of Individual Scores\]"
        scores_match = re.search(scores_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if not scores_match:
            return False, [], None, "Could not find individual scores section with format [The Begin of Individual Scores]"
        
        scores_str = scores_match.group(1).strip()
        scores = [s.strip() for s in scores_str.split(',')]
        
        # Validate score count
        if num_responses == 1 and len(scores) != 1:
            return False, [], None, f"Expected 1 score for single response, got {len(scores)}: {scores}"
        elif num_responses == 2 and len(scores) != 2:
            return False, [], None, f"Expected 2 scores for two responses, got {len(scores)}: {scores}"
        
        # Extract ranking if two responses
        ranking = None
        if num_responses == 2:
            ranking_pattern = r"\[The Begin of Ranking Score\]\s*\\boxed\{(\d+)\}\s*\[The End of Ranking Score\]"
            ranking_match = re.search(ranking_pattern, response, re.DOTALL | re.IGNORECASE)
            
            if not ranking_match:
                return False, scores, None, "Missing ranking score section with format [The Begin of Ranking Score]"
            
            ranking = ranking_match.group(1).strip()
            
            # Validate ranking is a single value
            if ',' in ranking:
                return False, scores, None, "Ranking should be a single value, found comma"
        
        return True, scores, ranking, ""
        
    except Exception as e:
        return False, [], None, f"Scoring parse error: {str(e)}"



# ========================= TWO-STAGE ENVIRONMENT =========================

@ray.remote
class TwoStageFactCheckEnvironment(EnvironmentInterface):
    """Two-stage fact-checking environment: Stage 1 fact-check, Stage 2 score."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.format_penalty = cfg.get("format_penalty", -20)
        self.factcheck_bonus_multiplier = cfg.get("factcheck_bonus_multiplier", 0.0)
        logging.basicConfig(level=logging.INFO)
    

    def step(self, message_log_batch: list[list[dict[str, str]]], metadata: list[TwoStageMetadata]) -> EnvironmentReturn:
        """Process two-stage fact-checking and scoring - debug version."""
        
        print(f"\n[TWO-STAGE ENV] Processing batch of {len(message_log_batch)} samples")
        
        rewards = []
        observations = []
        next_metadata = []
        
        for i, (conversation, meta) in enumerate(zip(message_log_batch, metadata)):
            if i < 2:  # First couple samples
                print(f"\n[SAMPLE {i}] Processing sample {i}")
                print(f"  factcheck_stage_complete: {meta.get('factcheck_stage_complete', False)}")
                print(f"  Has ground truth: h1={meta.get('helpfulness_1')}, h2={meta.get('helpfulness_2')}, pref={meta.get('preference_ranking')}")
                print("conversation: ", conversation)
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
            if not meta.get("factcheck_stage_complete"):
                # STAGE 1: Fact-checking
                reward, obs, updated_meta = self._process_factcheck_stage(
                    rm_response, meta
                )
                # CRITICAL: Ensure we move to next stage
                if updated_meta and not updated_meta.get("factcheck_stage_complete"):
                    print(f"[WARNING] Sample {i} didn't complete factcheck stage properly")
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
        '''
        print(f"\n[TWO-STAGE ENV] Batch summary:")
        print(f"  Rewards: {rewards}")
        print(f"  Terminated: {terminateds.tolist()}")
        print(f"  Non-zero rewards: {sum(1 for r in rewards if r != 0)}")
        '''

        # Add stop strings for each stage
        next_stop_strings = []
        for meta in next_metadata:
            if meta and not meta.get("factcheck_stage_complete"):
                # Fact-checking stage - stop after fact-check output
                next_stop_strings.append(["[End of Fact Checking for Response 2]"])
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

    def _process_factcheck_stage(self, response: str, metadata: TwoStageMetadata) -> Tuple[float, dict, TwoStageMetadata]:
        """Process fact-checking stage."""
        
        # Parse and store fact-check results
        is_parsed, parsed_response = parse_fact_checking_response(response)
        
        # Store fact-check results and prepare for scoring stage
        updated_metadata = metadata.copy()
        updated_metadata["factcheck_stage_complete"] = True
        updated_metadata["factcheck_results"] = parsed_response
        
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
        obs = {"role": "user", "content": "<|im_start|>user\n" + scoring_prompt + "<|im_end|>\n<|im_start|>assistant\n"}
        return reward, obs, updated_metadata



    def _process_scoring_stage(self, response: str, metadata: TwoStageMetadata) -> Tuple[float, dict, TwoStageMetadata]:
        """Process scoring stage with extensive debugging."""
        '''
        print(f"\n[SCORING STAGE DEBUG] Starting scoring stage")
        print(f"  Response length: {len(response)}")
        print(f"  Response preview: {response}...")
        print(f"  Metadata num_responses: {metadata['num_responses']}")
        print(f"  Metadata helpfulness_1: {metadata.get('helpfulness_1')}")
        print(f"  Metadata helpfulness_2: {metadata.get('helpfulness_2')}")
        print(f"  Metadata preference_ranking: {metadata.get('preference_ranking')}")
        '''
        # Parse scoring response (still need this for reward calculation)
        is_valid, scores, ranking, error_msg = parse_scoring_response(
            response, metadata["num_responses"]
        )
        '''
        print(f"  Parse results: is_valid={is_valid}")
        print(f"  Extracted scores: {scores}")
        print(f"  Extracted ranking: {ranking}")
        '''
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
        base_reward = 0.0
        reward_breakdown = []
        
        try:
            if metadata["num_responses"] == 1:
                if metadata["helpfulness_1"] is not None and scores and len(scores) > 0:
                    score_1 = int(scores[0])
                    distance_1 = abs(score_1 - metadata["helpfulness_1"])
                    base_reward = -distance_1
                    reward_breakdown.append(f"single_score_distance: -{distance_1}")
                    #print(f"  Single response: predicted={score_1}, ground_truth={metadata['helpfulness_1']}, distance={distance_1}")
                else:
                    print(f"  Single response: missing data - helpfulness_1={metadata.get('helpfulness_1')}, scores={scores}")
            
            elif metadata["num_responses"] == 2:
                if metadata["helpfulness_1"] is not None and metadata["helpfulness_2"] is not None and scores and len(scores) >= 2:
                    score_1 = int(scores[0])
                    score_2 = int(scores[1])
                    distance_1 = abs(score_1 - metadata["helpfulness_1"])
                    distance_2 = abs(score_2 - metadata["helpfulness_2"])
                    base_reward = -(distance_1 + distance_2)
                    reward_breakdown.append(f"score_distances: -{distance_1} + -{distance_2}")
                    #print(f"  Score 1: predicted={score_1}, ground_truth={metadata['helpfulness_1']}, distance={distance_1}")
                    #print(f"  Score 2: predicted={score_2}, ground_truth={metadata['helpfulness_2']}, distance={distance_2}")
                else:
                    print(f"  Two responses: missing score data - helpfulness_1={metadata.get('helpfulness_1')}, helpfulness_2={metadata.get('helpfulness_2')}, scores={scores}")
                
                if metadata["preference_ranking"] is not None and ranking:
                    try:
                        ranking_int = int(ranking)
                        ranking_distance = abs(ranking_int - metadata["preference_ranking"])
                        base_reward -= ranking_distance
                        reward_breakdown.append(f"ranking_distance: -{ranking_distance}")
                        #print(f"  Ranking: predicted={ranking_int}, ground_truth={metadata['preference_ranking']}, distance={ranking_distance}")
                    except ValueError as e:
                        print(f"  Ranking parse error: {e}, ranking='{ranking}'")
                else:
                    print(f"  Two responses: missing ranking data - preference_ranking={metadata.get('preference_ranking')}, ranking={ranking}")
                        
        except ValueError as e:
            print(f"[SCORING STAGE] Score parsing error: {e}")
            print(f"  Scores that failed to parse: {scores}")
            obs = {
                "role": "environment",
                "content": f"<environment>Score parsing error: {e}</environment>"
            }
            return float(self.format_penalty), obs, None

        
        # Final reward is just the base accuracy (no fact-checking bonus/penalty)
        final_reward = base_reward
        
        #print(f"  Reward breakdown: {', '.join(reward_breakdown)}")
        #print(f"  Final reward: {final_reward}")
        obs = {
            "role": "environment",
            "content": f"<environment>Two-stage completed. Final reward: {final_reward} (breakdown: {', '.join(reward_breakdown)})</environment>",
        }
        return float(final_reward), obs, None  # Terminate episode



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
    
    def shutdown(self):
        """Clean up resources."""
        pass

