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
from nemo_rl.environments.genrm_environment_base import distance_abs


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
    return "This function is no longer used - scoring prompt is created directly in _process_factcheck_stage"

# ========================= PARSING UTILITIES =========================

def parse_fact_checking_response(response: str, num_responses: int = 2) -> Tuple[bool, str]:
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

def filter_wrong_and_unknown_facts(factcheck_results: str) -> str:
    """
    Filter fact-checking results to only include Wrong and Unknown facts.
    
    Args:
        factcheck_results: The original fact-checking results
        
    Returns:
        Filtered fact-checking results containing only Wrong and Unknown facts
    """
    try:
        lines = factcheck_results.split('\n')
        filtered_lines = []
        current_response_block = []
        in_response_block = False
        current_response_num = None
        
        for line in lines:
            # Check if we're entering a fact checking block
            response_start_match = re.match(r'\[Fact Checking for Response (\d+)\]', line.strip())
            if response_start_match:
                in_response_block = True
                current_response_num = response_start_match.group(1)
                current_response_block = [line]
                continue
            
            # Check if we're exiting a fact checking block
            response_end_match = re.match(r'\[End of Fact Checking for Response (\d+)\]', line.strip())
            if response_end_match:
                in_response_block = False
                # Add the end marker
                current_response_block.append(line)
                
                # Filter the facts in this block and add to final result
                filtered_block = filter_facts_in_block(current_response_block, current_response_num)
                filtered_lines.extend(filtered_block)
                if filtered_lines and not filtered_lines[-1].strip():
                    # Don't add extra empty line
                    pass
                else:
                    filtered_lines.append('')  # Add spacing between response blocks
                
                current_response_block = []
                current_response_num = None
                continue
            
            # If we're inside a response block, collect lines
            if in_response_block:
                current_response_block.append(line)
            else:
                # Lines outside response blocks (like empty lines between blocks)
                continue
        
        # Join and clean up
        result = '\n'.join(filtered_lines).strip()
        return result if result else "No Wrong or Unknown facts identified."
        
    except Exception as e:
        print(f"Error filtering facts: {e}")
        return factcheck_results  # Return original if filtering fails

def filter_facts_in_block(block_lines: list[str], response_num: str) -> list[str]:
    """
    Filter facts within a single response block to keep only Wrong and Unknown facts.
    
    Args:
        block_lines: Lines within a fact checking block
        response_num: The response number for this block
        
    Returns:
        Filtered lines for this block
    """
    filtered_facts = []
    
    # Start with the block header
    result_lines = [block_lines[0]]  # [Fact Checking for Response X]
    
    # Process each line in the block (skip header and footer)
    for line in block_lines[1:-1]:  # Skip first (header) and last (footer)
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a fact line with the expected format
        # Pattern: (N) Factual Claim: ... | Status: ... | Correction: ...
        fact_match = re.match(r'\(\d+\)\s*Factual Claim:.*?\|\s*Status:\s*(Correct|Wrong|Unknown)', line, re.IGNORECASE)
        
        if fact_match:
            status = fact_match.group(1).strip().lower()
            if status in ['wrong', 'unknown']:
                filtered_facts.append(line)
        else:
            # If line doesn't match the expected format, include it anyway
            # (could be continuation of previous fact or other content)
            if filtered_facts:  # Only include if we have at least one filtered fact
                filtered_facts.append(line)
    
    # Add the filtered facts to result
    if filtered_facts:
        result_lines.extend(filtered_facts)
    else:
        # If no Wrong/Unknown facts found, add a note
        result_lines.append("No Wrong or Unknown facts identified.")
    
    # Add the block footer
    result_lines.append(block_lines[-1])  # [End of Fact Checking for Response X]
    
    return result_lines



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
class TwoStageFactCheckEnvironment(EnvironmentInterface):
    """Two-stage fact-checking environment: Stage 1 fact-check, Stage 2 score."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.format_penalty = cfg.get("format_penalty", -100)
        self.factcheck_bonus_multiplier = cfg.get("factcheck_bonus_multiplier", 0.0)
        logging.basicConfig(level=logging.INFO)
    

    def step(self, message_log_batch: list[list[dict[str, str]]], metadata: list[TwoStageMetadata]) -> EnvironmentReturn:
        """Process two-stage fact-checking and scoring."""
        
        print(f"\n[TWO-STAGE ENV] Processing batch of {len(message_log_batch)} samples")
        
        rewards = []
        observations = []
        next_metadata = []
        
        for i, (conversation, meta) in enumerate(zip(message_log_batch, metadata)):
            if i < 2:  # First couple samples
                print(f"\n[SAMPLE {i}] Processing sample {i}")
                print(f"  factcheck_stage_complete: {meta.get('factcheck_stage_complete', False)}")
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
        
        # Filter to only include Wrong and Unknown facts for the scoring stage
        if is_parsed:
            filtered_response = filter_wrong_and_unknown_facts(parsed_response)
            print(f"[FACT-CHECK FILTERING] Original length: {len(parsed_response)}, Filtered length: {len(filtered_response)}")
        else:
            filtered_response = "No valid fact checking results found."
        
        # Store ORIGINAL fact-check results in metadata (for potential future use)
        updated_metadata = metadata.copy()
        updated_metadata["factcheck_stage_complete"] = True
        updated_metadata["factcheck_results"] = parsed_response  # Store original
        
        # CRITICAL: Create scoring prompt that includes FILTERED factcheck results
        # This ensures only Wrong/Unknown facts are considered in scoring
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
        
        reward = 0.0 if is_parsed else float(self.format_penalty)

        # Return observation that becomes the next user message
        # This includes the filtered factcheck results directly in the scoring prompt
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


        
        # Final reward is the base accuracy
        final_reward = base_reward
        
        obs = {
            "role": "environment",
            "content": f"<environment>Two-stage completed. Final reward: {final_reward} (breakdown: {', '.join(reward_breakdown)})</environment>",
        }
        return float(final_reward), obs, None  # Terminate episode

    def global_post_process_and_metrics(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]:
        """Calculate metrics for filtered two-stage approach."""
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
            "approach": "filtered_two_stage",  # Indicate this is the filtered version
        }
        
        return batch, metrics
    
    def shutdown(self):
        """Clean up resources."""
        pass