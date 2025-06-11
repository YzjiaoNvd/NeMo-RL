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

            distance = 100
            error_details = []

            try:
                # Extract individual helpfulness scores
                individual_scores_match = re.search(
                    r'\[The Begin of Individual Scores\](.*?)\[The End of Individual Scores\]',
                    assistant_response,
                    re.DOTALL
                )
                
                if individual_scores_match:
                    scores_text = individual_scores_match.group(1)
                    extracted_scores = self.extract_answer(scores_text)
                    
                    if extracted_scores:
                        individual_scores = [s.strip() for s in extracted_scores.split(",")]
                        
                        if meta["num_responses"] == 1:
                            if meta["helpfulness_1"] is not None:
                                distance = self.distance_abs(individual_scores[0], meta["helpfulness_1"])
                        
                        elif meta["num_responses"] == 2:
                            if meta["helpfulness_1"] is not None and len(individual_scores) > 0:
                                distance = self.distance_abs(individual_scores[0], meta["helpfulness_1"])
                            if meta["helpfulness_2"] is not None and len(individual_scores) > 1:
                                distance += self.distance_abs(individual_scores[1], meta["helpfulness_2"])
                            
                            # Extract preference ranking
                            preference_match = re.search(
                                r'\[The Begin of Ranking Score\](.*?)\[The End of Ranking Score\]',
                                assistant_response,
                                re.DOTALL
                            )
                            
                            if preference_match and meta["preference_ranking"] is not None:
                                pref_text = preference_match.group(1)
                                preference_ranking = self.extract_answer(pref_text)
                                if preference_ranking:
                                    distance += self.distance_abs(preference_ranking, meta["preference_ranking"])

            except Exception as e:
                print(f"Error processing response: {e}")
                distance = 100

            # Calculate reward (negative distance)
            reward = -distance
            print("#################")
            print(f"Distance: {distance}, Errors: {error_details}")
            print(f"Metadata: {meta}")
            print(f"Full response: {assistant_response}")
            print("#################")

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