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
            retval = None
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
            try:
                # get individual helpfulness scores
                indidual_scores_paragraph = assistant_response.split("[The Begin of Individual Scores]")[-1].split("[The End of Individual Scores]")[0]
                individual_scores = self.extract_answer(indidual_scores_paragraph).split(",")
                if meta["num_responses"] == 1:
                    gt_individual = meta["helpfulness_1"] 
                    distance = self.distance_abs(individual_scores[0], gt_individual)
                elif meta["num_responses"] == 2:
                    gt_individual_1 = meta.get("helpfulness_1", None)
                    gt_individual_2 = meta.get("helpfulness_2", None)
                    distance = 0
                    if gt_individual_1 is not None and gt_individual_2 is not None:
                        distance = self.distance_abs(individual_scores[0], gt_individual_1) + self.distance_abs(individual_scores[1], gt_individual_2)

                    # get preference ranking score
                    preference_ranking_paragraph = assistant_response.split("[The Begin of Ranking Score]")[-1].split("[The End of Ranking Score]")[0]
                    preference_ranking = self.extract_answer(preference_ranking_paragraph)
                    gt_preference_ranking = meta["preference_ranking"]
                    distance += self.distance_abs(preference_ranking, gt_preference_ranking)

                else:
                    raise ValueError(f"Unsupported number of responses for genrm: {meta['num_responses']}")
                
                
            except Exception as e:
                logging.error(f"Error verifying response: {assistant_response}")
                logging.error(f"Error: {e}")
                distance = 100


            reward = -distance
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
    
    def global_post_process_and_metrics(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]:
        """Post processing and metrics calculation."""
        rewards = batch.get("rewards", torch.tensor([]))
        num_samples = len(batch.get("idx", []))
    
        if len(rewards) == 0:
            return batch, {}
    
        mean_reward = rewards.mean().item() if "rewards" in batch else 0.0
        perfect_pred = (rewards == 0).float().mean().item()  # Exact matches

        metrics = {
            "mean_reward": mean_reward,
            "perfect_pred_rate": perfect_pred,
            "num_samples": num_samples,
        }
        
        return batch, metrics