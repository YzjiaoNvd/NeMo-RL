# nemo_rl/environments/genrm_environment.py
import re
import logging
from typing import Any, Optional, TypedDict

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
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            return None
        else:
            retval = string[idx : right_brace_idx + 1]

        if retval:
            left = "\\boxed{"
            try:
                assert retval[: len(left)] == left
                assert retval[-1] == "}"
                return retval[len(left) : -1]
            except AssertionError:
                return None

        return None

    def distance_abs(self, a: str, b: int) -> int:
        """Calculate absolute distance between predicted and ground truth."""
        try:
            d = abs(int(a) - int(b))
        except Exception as e:
            logging.error(f"Error calculating distance: {e}, a: {a}, b: {b}")
            d = 100
        return d
    
    def step(
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[GenRMEnvironmentMetadata],
    ) -> EnvironmentReturn:
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
                                distance += self.distance_abs(individual_scores[0], meta["helpfulness_1"])
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
                
                reward = -distance
                
                
            except Exception as e:
                logging.error(f"Error processing response: {e}")
                reward = -100
            
            print("Metadata: ", meta)
            print("Reward: ", reward)
            print("assistant_response: ", assistant_response)
            
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
    
    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        """Post processing and metrics calculation."""
        mean_reward = batch["rewards"].mean().item() if "rewards" in batch else 0.0
        positive_rewards = (batch["rewards"] > -10).float().mean().item() if "rewards" in batch else 0.0
        
        metrics = {
            "mean_reward": mean_reward,
            "positive_reward_rate": positive_rewards,
            "num_samples": len(batch.get("idx", [])),
        }
        
        return batch, metrics
