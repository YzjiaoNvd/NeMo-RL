import os
import json
import re
from collections import defaultdict

def calculate_accuracy(data: list[dict]):
    correct_predictions = 0
    total_predictions = 0

    # Iterate over each record in the file
    for record in data:
        total_predictions += 1
        # Safely get values from nested dictionaries
        predicted_ranking = record.get('predicted_ranking')
        metadata = record.get('metadata', {})
        preference_ranking = metadata.get('preference_ranking')

        # Ensure the required keys exist
        if predicted_ranking is not None and preference_ranking is not None:
            # Apply the mapping rule
            # 0 if predicted_ranking <= 3 else 1
            mapped_prediction = 0 if predicted_ranking <= 3 else 1
                            
            # Check if the prediction is correct
            if mapped_prediction == preference_ranking:
                correct_predictions += 1           
        else:
            print(f"    [WARNING] A record is missing 'predicted_ranking' or 'preference_ranking', skipping.")

    # Calculate accuracy, avoiding division by zero
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        return accuracy
    else:
        print(f"    [WARNING] No valid data found in the file.")
        return 0.0
            


def calculate_step_accuracies(directory_path: str, dataset: str) -> dict:
    """
    Calculates the accuracy for all step_*_judgebench_results.json files in a directory.

    The accuracy is calculated as follows:
    - Prediction: 0 if 'predicted_ranking' <= 3, else 1.
    - Ground Truth: 'metadata' -> 'preference_ranking'.
    - Accuracy = (Number of correct predictions) / (Total samples).

    Args:
        directory_path: The path to the directory containing the JSON files.

    Returns:
        A dictionary where keys are the step numbers (int) and values are the
        corresponding accuracies (float).
    """
    step_accuracies = {}

    # Regex to match filenames and extract the step number
    # e.g., "step_5_judgebench_results.json" -> "5"
    file_pattern = re.compile(rf'step_(\d+)_{dataset}_results\.json')
    print(f"Scanning directory: '{directory_path}'...")

    try:
        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            match = file_pattern.match(filename)
            # If the filename matches our pattern
            if match:
                step_number = int(match.group(1))
                file_path = os.path.join(directory_path, filename)
                
                print(f"  Processing file: {filename} (Step: {step_number})")

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if not isinstance(data, list):
                        print(f"    [WARNING] Content of {filename} is not a list, skipping.")
                        continue
                    
                    accuracy = 0.0
                    if dataset == "rewardbench2":
                        chunk_outcomes = []
                        if data:
                            length = len(data)
                            assert length % 3 == 0
                            for i in range(0, length, 3):
                                chunk = data[i: i+3]
                                if chunk:
                                    chunk_accuracy = calculate_accuracy(chunk)
                                    chunk_outcomes.append(1 if chunk_accuracy == 1.0 else 0)
                        
                        # The final accuracy is the average of the outcomes of all chunks
                        if chunk_outcomes:
                            accuracy = sum(chunk_outcomes) / len(chunk_outcomes)
                        else:
                            accuracy = 0.0 # No chunks to process
                    else:
                        # Use the standard accuracy calculation for all other datasets
                        accuracy = calculate_accuracy(data)

                    step_accuracies[step_number] = accuracy
                    
                except json.JSONDecodeError:
                    print(f"    [ERROR] File {filename} is not a valid JSON, skipping.")
                except Exception as e:
                    print(f"    [ERROR] An unknown error occurred while processing {filename}: {e}")

    except FileNotFoundError:
        print(f"[ERROR] Directory not found: '{directory_path}'")
        return {}

    # Sort the results by step number for clear presentation
    sorted_accuracies = dict(sorted(step_accuracies.items()))
    
    return sorted_accuracies



dataset = "rmbench"
data_directory = '/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/results/grpo_hs3_16K_step240_clip_max_0.28_qwen25_3b_lr_2e-6_temp_1_kl_0.001_grpo_bs_256_rollout_16_num_prompts_128/outputs'
final_accuracies = calculate_step_accuracies(data_directory, dataset)

if final_accuracies:
    print("Accuracy for each step:")
    for step, acc in final_accuracies.items():
        # Use :.2% to format as a percentage
        print(f"  Step {step}: {acc:.2%}")
else:
    print("No matching files or data were found in the specified directory.")