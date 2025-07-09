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
        self.format_penalty = cfg.get("format_penalty", -100)  # Large penalty for format violations
        
        # Reward design selection
        # Options: "current", "strict_format", "squared_error", "decomposed"
        self.reward_design = cfg.get("reward_design")
        print(self.reward_design)
        # self.reward_design = "current"

        # Map reward designs to their corresponding functions
        self.reward_functions = {
            "r0": self._calculate_current_reward,
            "r1": self._calculate_strict_format_reward,
            "r2": self._calculate_squared_error_reward,
            "r3": self._calculate_decomposed_reward,
        }
        
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Using reward design: {self.reward_design}")
    
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

    def distance_squared(self, predicted: Any, ground_truth: Any) -> int:
        """Calculate squared distance between predicted and ground truth."""
        abs_dist = self.distance_abs(predicted, ground_truth)
        return abs_dist ** 2

    def validate_format_and_extract(self, assistant_response: str, num_responses: int, 
                                   strict_mode: bool = False) -> tuple[bool, Optional[list], Optional[str], str]:
        """
        Validate the format and extract scores.
        
        Args:
            assistant_response: The model's response
            num_responses: Number of responses to evaluate
            strict_mode: If True, reject responses with content after the expected pattern
            
        Returns: (is_valid_format, individual_scores, preference_ranking, error_message)
        """
        if num_responses == 1:
            # Expected format: "[The Begin of Individual Scores]\n\\boxed{x} \n[The End of Individual Scores]\n"
            pattern = r'\[The Begin of Individual Scores\]\s*\n\s*\\boxed\{([^}]+)\}\s*\n\s*\[The End of Individual Scores\]'
            match = re.search(pattern, assistant_response, re.DOTALL)
            
            if not match:
                return False, None, None, "Format violation: Expected '[The Begin of Individual Scores]\\n\\\\boxed{x} \\n[The End of Individual Scores]\\n' for single response"
            
            score_content = match.group(1).strip()
            # For single response, should be just one score
            if ',' in score_content:
                return False, None, None, "Format violation: Single response should contain only one score, found comma"
            
            # For strict format, check if there's content after the pattern
            if strict_mode:
                end_pos = match.end()
                remaining_content = assistant_response[end_pos:].strip()
                if remaining_content:
                    return False, None, None, f"Format violation: Found additional content after [The End of Individual Scores]: '{remaining_content[:50]}...'"
            
            return True, [score_content], None, ""
            
        elif num_responses == 2:
            # Expected format: "[The Begin of Individual Scores]\n\\boxed{x, y} \n[The End of Individual Scores]\n[The Begin of Ranking Score]\n\\boxed{z} \n[The End of Ranking Score]"
            
            # Check individual scores section
            individual_pattern = r'\[The Begin of Individual Scores\]\s*\n\s*\\boxed\{([^}]+)\}\s*\n\s*\[The End of Individual Scores\]'
            individual_match = re.search(individual_pattern, assistant_response, re.DOTALL)
            
            if not individual_match:
                return False, None, None, "Format violation: Missing or incorrect '[The Begin of Individual Scores]\\n\\\\boxed{x, y} \\n[The End of Individual Scores]' section"
            
            # Check ranking score section
            ranking_pattern = r'\[The Begin of Ranking Score\]\s*\n\s*\\boxed\{([^}]+)\}\s*\n\s*\[The End of Ranking Score\]'
            ranking_match = re.search(ranking_pattern, assistant_response, re.DOTALL)
            
            if not ranking_match:
                return False, None, None, "Format violation: Missing or incorrect '[The Begin of Ranking Score]\\n\\\\boxed{z} \\n[The End of Ranking Score]' section"
            
            # Extract individual scores
            scores_content = individual_match.group(1).strip()
            individual_scores = [score.strip() for score in scores_content.split(',')]
            
            # For two responses, should have exactly two scores
            if len(individual_scores) != 2:
                return False, None, None, f"Format violation: Expected exactly 2 individual scores, found {len(individual_scores)}"
            
            # Extract preference ranking
            preference_content = ranking_match.group(1).strip()
            # Preference ranking should be a single value
            if ',' in preference_content:
                return False, None, None, "Format violation: Preference ranking should be a single value, found comma"
            
            # For strict format, check if there's content after the ranking pattern
            if strict_mode:
                end_pos = ranking_match.end()
                remaining_content = assistant_response[end_pos:].strip()
                if remaining_content:
                    return False, None, None, f"Format violation: Found additional content after [The End of Ranking Score]: '{remaining_content[:50]}...'"
            
            return True, individual_scores, preference_content, ""
        
        else:
            return False, None, None, f"Unsupported number of responses: {num_responses}"

    def _calculate_current_reward(self, individual_scores: list, preference_ranking: Optional[str], 
                                 meta: GenRMEnvironmentMetadata) -> float:
        """Calculate reward using current design (absolute distance)."""
        distance = 0
        
        if meta["num_responses"] == 1:
            if meta["helpfulness_1"] is not None and individual_scores:
                distance = self.distance_abs(individual_scores[0], meta["helpfulness_1"])
        elif meta["num_responses"] == 2:
            if meta["helpfulness_1"] is not None and individual_scores and len(individual_scores) >= 2:
                distance += self.distance_abs(individual_scores[0], meta["helpfulness_1"])
                distance += self.distance_abs(individual_scores[1], meta["helpfulness_2"])
            if meta["preference_ranking"] is not None and preference_ranking:
                distance += self.distance_abs(preference_ranking, meta["preference_ranking"])
        
        return -distance

    def _calculate_strict_format_reward(self, individual_scores: list, preference_ranking: Optional[str], 
                                       meta: GenRMEnvironmentMetadata) -> float:
        """Calculate reward using strict format design (same as current but with strict validation)."""
        # Same calculation as current reward
        return self._calculate_current_reward(individual_scores, preference_ranking, meta)

    def _calculate_squared_error_reward(self, individual_scores: list, preference_ranking: Optional[str], 
                                       meta: GenRMEnvironmentMetadata) -> float:
        """Calculate reward using squared error design."""
        distance = 0
        
        if meta["num_responses"] == 1:
            if meta["helpfulness_1"] is not None and individual_scores:
                distance = self.distance_squared(individual_scores[0], meta["helpfulness_1"])
        elif meta["num_responses"] == 2:
            if meta["helpfulness_1"] is not None and individual_scores and len(individual_scores) >= 2:
                distance += self.distance_squared(individual_scores[0], meta["helpfulness_1"])
                distance += self.distance_squared(individual_scores[1], meta["helpfulness_2"])
            if meta["preference_ranking"] is not None and preference_ranking:
                distance += self.distance_squared(preference_ranking, meta["preference_ranking"])
        
        return -distance

    def _calculate_decomposed_reward(self, individual_scores: list, preference_ranking: Optional[str], 
                                    meta: GenRMEnvironmentMetadata) -> float:
        """Calculate reward using decomposed design."""
        if meta["num_responses"] == 1:
            # For single response, use absolute distance
            distance = 0
            if meta["helpfulness_1"] is not None and individual_scores:
                distance = self.distance_abs(individual_scores[0], meta["helpfulness_1"])
            return -distance
        else:
            # Calculate individual score rewards
            score_reward = 0
            if meta["helpfulness_1"] is not None and individual_scores and len(individual_scores) >= 2:
                score_reward -= self.distance_abs(individual_scores[0], meta["helpfulness_1"])
                score_reward -= self.distance_abs(individual_scores[1], meta["helpfulness_2"])
            
            # Calculate decomposed ranking components
            if meta["preference_ranking"] is not None and preference_ranking:
                pred_rank = self.parse_score_value(preference_ranking)
                if pred_rank is not None:
                    components = self._get_decomposed_components(
                        pred_rank, 
                        meta["preference_ranking"],
                        individual_scores,
                        [meta["helpfulness_1"], meta["helpfulness_2"]]
                    )
                    
                    # Combine components with weights
                    ranking_reward = (
                        10.0 * components['correct_best'] +
                        5.0 * components['correct_preference'] +
                        2.0 * components['consistency']
                    )
                    
                    # Log components for debugging
                    logging.info(f"Decomposed components: {components}")
                    
                    return score_reward + ranking_reward
                else:
                    return self.format_penalty
            else:
                return score_reward

    def _get_decomposed_components(self, predicted_ranking: int, ground_truth_ranking: int, 
                                  predicted_scores: list, ground_truth_scores: list) -> dict:
        """Calculate individual components for decomposed ranking reward."""
        components = {}
        
        # Component 1: Correct identification of best response (0 or 1)
        # Rankings 1,2,3 mean first response is better; 4,5,6 mean second is better
        predicted_best = 0 if predicted_ranking <= 3 else 1
        true_best = 0 if ground_truth_ranking <= 3 else 1
        components['correct_best'] = 1.0 if predicted_best == true_best else 0.0
        
        # Component 2: Correct preference strength (0 or 1)
        components['correct_preference'] = 1.0 if predicted_ranking == ground_truth_ranking else 0.0
        
        # Component 3: Consistency between preference and score differences (0 or 1)
        if len(predicted_scores) == 2 and len(ground_truth_scores) == 2:
            try:
                pred_score_1 = self.parse_score_value(predicted_scores[0])
                pred_score_2 = self.parse_score_value(predicted_scores[1])
                
                if pred_score_1 is not None and pred_score_2 is not None:
                    score_diff = abs(pred_score_1 - pred_score_2)
                    
                    # Map preference rankings to strength categories
                    # 3,4: slight preference (diff should be <= 1)
                    # 2,5: medium preference (diff should be 1 or 2)
                    # 1,6: strong preference (diff should be >= 2)
                    
                    if predicted_ranking in [3, 4]:  # Slight preference
                        consistent = score_diff <= 1
                    elif predicted_ranking in [2, 5]:  # Medium preference
                        consistent = 1 <= score_diff <= 2
                    elif predicted_ranking in [1, 6]:  # Strong preference
                        consistent = score_diff >= 2
                    else:
                        consistent = False
                    
                    components['consistency'] = 1.0 if consistent else 0.0
                else:
                    components['consistency'] = 0.0
            except:
                components['consistency'] = 0.0
        else:
            components['consistency'] = 0.0
        
        return components
    
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
            
            # Use strict mode for strict_format design
            strict_mode = (self.reward_design == "strict_format")
            
            is_valid_format, individual_scores, preference_ranking, error_message = self.validate_format_and_extract(
                assistant_response, meta["num_responses"], strict_mode=strict_mode
            )
            
            if not is_valid_format:
                # Apply format penalty
                reward = self.format_penalty
                rewards.append(float(reward))
                observations.append({
                    "role": "environment",
                    "content": f"Format violation penalty applied. Error: {error_message}. Reward: {reward}"
                })
                
            else:
                # Select appropriate reward function
                if self.reward_design not in self.reward_functions:
                    raise ValueError(f"Unknown reward design: {self.reward_design}")
                
                reward_fn = self.reward_functions[self.reward_design]
                reward = reward_fn(individual_scores, preference_ranking, meta)

                rewards.append(float(reward))
                observations.append({
                    "role": "environment",
                    "content": f"Format correct. Reward design: {self.reward_design}. Reward: {reward}"
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
        
        # Calculate format violation rate (rewards equal to format_penalty)
        format_violation_rate = float(np.mean(rewards_np == self.format_penalty))
        
        # For non-format-penalty rewards, calculate accuracy metrics
        accuracy_rewards = rewards_np[rewards_np != self.format_penalty]
        if len(accuracy_rewards) > 0:
            mean_accuracy_reward = float(np.mean(accuracy_rewards))
            
            # Different thresholds for different reward designs
            if self.reward_design == "squared_error":
                # For squared error, perfect is 0, good might be <= -25 (5^2)
                perfect_pred = float(np.mean(accuracy_rewards == 0))
                good_pred = float(np.mean(accuracy_rewards >= -25))
            elif self.reward_design == "decomposed":
                # For decomposed, positive rewards are possible
                perfect_pred = float(np.mean(accuracy_rewards >= 17))  # All components perfect
                good_pred = float(np.mean(accuracy_rewards >= 0))     # At least break even
            else:
                # For absolute distance designs
                perfect_pred = float(np.mean(accuracy_rewards == 0))
                good_pred = float(np.mean(accuracy_rewards >= -10))
        else:
            mean_accuracy_reward = 0.0
            perfect_pred = 0.0
            good_pred = 0.0
        
        metrics = {
            "mean_reward": mean_reward,
            "mean_distance": -mean_reward if mean_reward != self.format_penalty else 0,
            "format_violation_rate": format_violation_rate,
            "mean_accuracy_reward": mean_accuracy_reward,
            "perfect_pred_rate": perfect_pred,
            "good_pred_rate": good_pred,
            "num_samples": num_samples,
            "format_correct_samples": len(accuracy_rewards),
            "reward_design": self.reward_design,
        }
        
        return batch, metrics
    
    def shutdown(self):
        """Clean up resources."""
        pass