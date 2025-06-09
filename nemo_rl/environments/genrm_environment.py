# nemo_rl/environments/genrm_environment.py
import re
import logging
from typing import Any, Optional, TypedDict
import numpy as np
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)

class GenRMEnvironmentMetadata(TypedDict):
    num_responses: int
    helpfulness_1: Optional[int]
    helpfulness_2: Optional[int]
    preference_ranking: Optional[int]


@ray.remote
class GenRMEnvironment(EnvironmentInterface):
    """Generative Reward Model environment for HelpSteer3 dataset."""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        logging.basicConfig(level=logging.INFO)
    
    def extract_answer(self, string: str) -> Optional[str]:
        """Extract Answer String from \\boxed expression."""
        # Try to find the last occurrence of \boxed
        idx = string.rfind("\\boxed")
        if idx < 0:
            # Try \fbox as alternative
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        # Find the matching braces
        i = idx + 6  # Skip past "\boxed"
        if i >= len(string) or string[i] != '{':
            return None
            
        # Count braces to find the matching closing brace
        brace_count = 0
        start_idx = i
        while i < len(string):
            if string[i] == '{':
                brace_count += 1
            elif string[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Extract content between braces
                    return string[start_idx + 1:i].strip()
            i += 1

        return None

    def parse_score_value(self, score_str: str) -> Optional[int]:
        """Parse a score value from a string, handling various formats."""
        if not score_str:
            return None
            
        score_str = score_str.strip()
        
        # Try direct conversion first
        try:
            return int(score_str)
        except ValueError:
            pass
        
        # Try to extract the first number found
        match = re.search(r'(\d+)', score_str)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
                
        return None

    def distance_abs(self, predicted: Any, ground_truth: Any) -> int:
        """Calculate absolute distance between predicted and ground truth."""
        try:
            # Handle None values
            if predicted is None or ground_truth is None:
                return 100
            
            # Convert to integers if they're strings
            if isinstance(predicted, str):
                predicted = self.parse_score_value(predicted)
                if predicted is None:
                    return 100
                    
            if isinstance(ground_truth, str):
                ground_truth = self.parse_score_value(ground_truth)
                if ground_truth is None:
                    return 100
            
            # Calculate distance
            return abs(int(predicted) - int(ground_truth))
        except Exception as e:
            logging.error(f"Error calculating distance: {e}, predicted: {predicted}, ground_truth: {ground_truth}")
            return 100
    
    def step(self, message_log_batch: list[list[dict[str, str]]], metadata: list[GenRMEnvironmentMetadata],) -> EnvironmentReturn:
        """Evaluate GenRM predictions and return rewards."""
        
        rewards = []
        observations = []
        
        for conversation, meta in zip(message_log_batch, metadata):
            # Extract assistant's response
            assistant_response = ""
            for msg in conversation:
                if msg["role"] == "assistant":
                    assistant_response = msg["content"]
                    break

            distance = 0
            error_details = []
            
            try:
                # Check if response contains expected sections
                has_individual_scores = "[The Begin of Individual Scores]" in assistant_response and "[The End of Individual Scores]" in assistant_response
                has_ranking_score = "[The Begin of Ranking Score]" in assistant_response and "[The End of Ranking Score]" in assistant_response
                
                if not has_individual_scores:
                    error_details.append("Missing individual scores section")
                    distance += 50
                else:
                    # Extract individual helpfulness scores
                    individual_scores_section = assistant_response.split("[The Begin of Individual Scores]")[1].split("[The End of Individual Scores]")[0]
                    scores_text = self.extract_answer(individual_scores_section)
                    
                    if scores_text is None:
                        error_details.append("Failed to extract individual scores from boxed format")
                        distance += 50
                    else:
                        # Parse individual scores
                        score_parts = [s.strip() for s in scores_text.split(",")]
                        individual_scores = []
                        
                        for i, score_str in enumerate(score_parts):
                            score = self.parse_score_value(score_str)
                            if score is None:
                                error_details.append(f"Failed to parse score {i+1}: '{score_str}'")
                                distance += 25
                                individual_scores.append(0)
                            else:
                                individual_scores.append(score)
                        
                        # Compare with ground truth
                        if meta["num_responses"] == 1:
                            if len(individual_scores) >= 1 and meta.get("helpfulness_1") is not None:
                                distance += self.distance_abs(individual_scores[0], meta["helpfulness_1"])
                            else:
                                distance += 50
                                
                        elif meta["num_responses"] == 2:
                            if len(individual_scores) >= 2:
                                if meta.get("helpfulness_1") is not None:
                                    distance += self.distance_abs(individual_scores[0], meta["helpfulness_1"])
                                if meta.get("helpfulness_2") is not None:
                                    distance += self.distance_abs(individual_scores[1], meta["helpfulness_2"])
                            else:
                                error_details.append(f"Expected 2 scores but got {len(individual_scores)}")
                                distance += 50
                
                # Handle ranking score for 2-response cases
                if meta["num_responses"] == 2:
                    if not has_ranking_score:
                        error_details.append("Missing ranking score section")
                        distance += 50
                    else:
                        # Extract preference ranking score
                        ranking_section = assistant_response.split("[The Begin of Ranking Score]")[1].split("[The End of Ranking Score]")[0]
                        ranking_text = self.extract_answer(ranking_section)
                        
                        if ranking_text is None:
                            error_details.append("Failed to extract ranking score from boxed format")
                            distance += 50
                        else:
                            ranking_score = self.parse_score_value(ranking_text)
                            if ranking_score is None:
                                error_details.append(f"Failed to parse ranking score: '{ranking_text}'")
                                distance += 50
                            elif meta.get("preference_ranking") is not None:
                                distance += self.distance_abs(ranking_score, meta["preference_ranking"])
                
            except Exception as e:
                logging.error(f"Error processing response: {str(e)}")
                logging.error(f"Response content: {assistant_response[:500]}...")  # Log first 500 chars
                error_details.append(f"Exception: {str(e)}")
                distance = 100

            # Calculate reward (negative distance)
            reward = -distance
            
            # Debug logging
            if distance > 0:
                logging.debug(f"Distance: {distance}, Errors: {error_details}")
                logging.debug(f"Metadata: {meta}")
                if len(assistant_response) < 1000:
                    logging.debug(f"Full response: {assistant_response}")

            rewards.append(float(reward))
            observations.append({
                "role": "environment",
                "content": f"GenRM evaluation complete. Distance: {distance}, Reward: {reward}"
            })
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminateds = torch.ones_like(rewards_tensor, dtype=torch.bool)
        
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards_tensor,
            terminateds=terminateds,
        )
    
    def global_post_process_and_metrics(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]:
        """Post processing and metrics calculation."""
        rewards = batch.get("rewards", torch.tensor([]))
        num_samples = len(batch.get("idx", []))
        
        if len(rewards) == 0:
            return batch, {}
        
        # Convert rewards to numpy for easier computation
        rewards_np = rewards.numpy() if hasattr(rewards, 'numpy') else np.array(rewards)
        
        # Calculate metrics
        mean_reward = float(np.mean(rewards_np))
        perfect_pred = float(np.mean(rewards_np == 0))  # Distance 0 = perfect
        good_pred = float(np.mean(rewards_np >= -10))  # Distance <= 10 = good
        
        metrics = {
            "mean_reward": mean_reward,
            "mean_distance": -mean_reward,  # Since reward = -distance
            "perfect_pred_rate": perfect_pred,
            "good_pred_rate": good_pred,
            "num_samples": num_samples,
        }
        
        return batch, metrics
    
    def shutdown(self):
        """Clean up resources."""
        pass