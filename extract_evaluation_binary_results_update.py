import os
import json
import re
from collections import defaultdict
import argparse

def calculate_accuracy_by_sample_and_domain(data: list[dict]):
    """
    Calculate accuracy grouped by sample_id and also by domain.
    An example (group with same sample_id) is correct only if ALL predictions are correct.
    
    Returns:
        tuple: (overall_accuracy, domain_accuracies, sample_count, domain_sample_counts)
    """
    # Group data by sample_id
    samples_by_id = defaultdict(list)
    
    for record in data:
        metadata = record.get('metadata')
        sample_id = metadata.get('sample_id')
        domain = metadata.get('domain')
        group_id = domain + "_" + str(sample_id)
        samples_by_id[group_id].append(record)
    
    if not samples_by_id:
        return 0.0, {}, 0, {}
    
    # Calculate accuracy for each sample and track by domain
    domain_results = defaultdict(list)  # domain -> list of sample results (0 or 1)
    sample_results = []
    
    for group_id, records in samples_by_id.items():
        # Check if all predictions in this sample are correct
        all_correct = True
        sample_domain = None
        
        for record in records:
            predicted_ranking = record.get('predicted_ranking')
            metadata = record.get('metadata')
            preference = metadata.get('preference')
            
            # Try to find domain in different possible locations
            sample_domain = metadata.get('domain') 

            if predicted_ranking != preference+1:
                all_correct = False
                # Don't break here - we want to check for domain consistency
        
        # Record result for this sample
        sample_result = 1.0 if all_correct else 0.0
        sample_results.append(sample_result)
        
        if sample_domain is not None:
            domain_results[sample_domain].append(sample_result)
    
    # Calculate overall accuracy
    overall_accuracy = sum(sample_results) / len(sample_results) if sample_results else 0.0
    
    # Calculate accuracy for each domain
    domain_accuracies = {}
    domain_sample_counts = {}
    for domain, results in domain_results.items():
        domain_accuracies[domain] = sum(results) / len(results) if results else 0.0
        domain_sample_counts[domain] = len(results)
    
    return overall_accuracy, domain_accuracies, len(sample_results), domain_sample_counts

def calculate_step_accuracies(directory_path: str, dataset: str) -> dict:
    """
    Calculates the accuracy for all step_*_results.json files in a directory.
    Groups by sample_id and calculates accuracy per domain.
    """
    step_results = {}

    # Regex to match filenames and extract the step number
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
                        original_data = json.load(f)

                    if not isinstance(original_data, list):
                        print(f"    [WARNING] Content of {filename} is not a list, skipping.")
                        continue
                    
                    data = original_data

                    if data:
                        overall_accuracy, domain_accuracies, total_samples, domain_sample_counts = calculate_accuracy_by_sample_and_domain(data)
                        step_results[step_number] = {
                            'overall_accuracy': overall_accuracy,
                            'domain_accuracies': domain_accuracies,
                            'total_samples': total_samples,
                            'domain_sample_counts': domain_sample_counts,
                            'total_data_points': len(data)
                        }
                        
                        print(f"    Processed {len(data)} data points grouped into {total_samples} samples")
                        
                    else:
                        step_results[step_number] = {
                            'overall_accuracy': 0.0,
                            'domain_accuracies': {},
                            'total_samples': 0,
                            'domain_sample_counts': {},
                            'total_data_points': 0
                        }
                    
                except json.JSONDecodeError:
                    print(f"    [ERROR] File {filename} is not a valid JSON, skipping.")
                except Exception as e:
                    print(f"    [ERROR] An unknown error occurred while processing {filename}: {e}")

    except FileNotFoundError:
        print(f"[ERROR] Directory not found: '{directory_path}'")
        return {}

    # Sort the results by step number for clear presentation
    sorted_results = dict(sorted(step_results.items()))
    
    return sorted_results


parser = argparse.ArgumentParser(description="Compute per-step accuracies from evaluation JSON files.")
parser.add_argument(
    "path",
    help="Path to the directory that contains the evaluation output files."
)
parser.add_argument("--dataset", default="rmbench", help="Dataset name (default: %(default)s).")

args = parser.parse_args()

final_results = calculate_step_accuracies(args.path, args.dataset)

if final_results:
    print("\nResults for each step:")
    print("=" * 50)
    
    for step, results in final_results.items():
        overall_acc = results['overall_accuracy']
        domain_accs = results['domain_accuracies']
        total_samples = results['total_samples']
        domain_counts = results['domain_sample_counts']
        total_data_points = results['total_data_points']
        
        print(f"\nStep {step}:")
        print(f"  Overall Accuracy: {overall_acc:.2%} ({total_samples} samples, {total_data_points} data points)")
        
        if domain_accs:
            print(f"  Domain Accuracies:")
            for domain, acc in sorted(domain_accs.items()):
                sample_count = domain_counts.get(domain)
                print(f"    {domain}: {acc:.2%} ({sample_count} samples)")
        else:
            print(f"  No domain information found")
            
else:
    print("No matching files or data were found in the specified directory.")