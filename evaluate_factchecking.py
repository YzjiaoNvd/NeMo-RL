import json
import os
import glob
import re
from pathlib import Path

def evaluate_factchecking(rmbench_file, chat_filtered_file):
    """
    Evaluate factchecking performance by comparing error_keys from chat_filtered.json
    with factcheck_response content in rmbench_results.json
    
    Args:
        rmbench_file (str): Path to rmbench_results.json
        chat_filtered_file (str): Path to chat_filtered.json
    
    Returns:
        dict: Dictionary containing evaluation results
    """
    
    # Load the JSON files
    try:
        with open(rmbench_file, 'r', encoding='utf-8') as f:
            rmbench_data = json.load(f)
        
        with open(chat_filtered_file, 'r', encoding='utf-8') as f:
            chat_filtered_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return None
    
    # Create a mapping from id to chat_filtered instances for quick lookup
    chat_filtered_dict = {item['id']: item for item in chat_filtered_data}
    
    hits = 0
    total_valid_instances = 0
    detailed_results = []
    
    # Process each instance in rmbench_results
    for instance_a in rmbench_data:
        # Check if this instance has the required fields
        if 'metadata' not in instance_a or 'sample_id' not in instance_a['metadata']:
            continue
            
        sample_id = instance_a['metadata']['sample_id']
        
        # Check if sample_id follows the "chat/XXX" pattern
        if not sample_id.startswith('chat/'):
            continue
            
        # Extract the XXX part
        try:
            chat_id = int(sample_id.split('chat/')[1])
        except (IndexError, ValueError):
            continue
            
        # Find corresponding instance B in chat_filtered
        if chat_id not in chat_filtered_dict:
            continue
            
        instance_b = chat_filtered_dict[chat_id]
        total_valid_instances += 1
        
        # Check if error_key exists in factcheck_response
        factcheck_response = instance_a.get('factcheck_response')
        error_key = instance_b.get('error_key')
        
        # Check if error_key appears in factcheck_response
        is_hit = error_key.lower() in factcheck_response.lower() if error_key else False
        
        if is_hit:
            hits += 1
            
        # Store detailed results for analysis
        detailed_results.append({
            'sample_id': sample_id,
            'chat_id': chat_id,
            'error_key': error_key,
            'hit': is_hit,
            'factcheck_response_snippet': factcheck_response[:500] + '...' if len(factcheck_response) > 500 else factcheck_response
        })
    
    # Calculate hit rate
    hit_rate = hits / total_valid_instances if total_valid_instances > 0 else 0
    
    results = {
        'total_valid_instances': total_valid_instances,
        'hits': hits,
        'hit_rate': hit_rate,
        'hit_rate_percentage': hit_rate * 100,
        'detailed_results': detailed_results
    }
    
    return results

def print_evaluation_results(results):
    """
    Print formatted evaluation results
    """
    if results is None:
        return
        
    print("=" * 50)
    print("FACTCHECKING EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total valid instances: {results['total_valid_instances']}")
    print(f"Hits: {results['hits']}")
    print(f"Hit rate: {results['hit_rate']:.4f}")
    print(f"Hit rate percentage: {results['hit_rate_percentage']:.2f}%")
    print()
    
    # Print some examples
    print("SAMPLE RESULTS:")
    print("-" * 30)
    for i, detail in enumerate(results['detailed_results'][:5]):  # Show first 5 examples
        print(f"Example {i+1}:")
        print(f"  Sample ID: {detail['sample_id']}")
        print(f"  Error key: '{detail['error_key']}'")
        print(f"  Hit: {detail['hit']}")
        print(f"  Factcheck snippet: {detail['factcheck_response_snippet']}")
        print()

def process_folder(folder_path, chat_filtered_file):
    """
    Process all step_id_rmbench_results.json files in a folder
    
    Args:
        folder_path (str): Path to folder containing the rmbench files
        chat_filtered_file (str): Path to chat_filtered.json
    
    Returns:
        dict: Dictionary containing results for all files
    """
    
    # Check if chat_filtered file exists
    if not os.path.exists(chat_filtered_file):
        print(f"Error: {chat_filtered_file} not found")
        return None
    
    # Find all files matching the pattern
    pattern = os.path.join(folder_path, "*_rmbench_results.json")
    rmbench_files = glob.glob(pattern)
    
    if not rmbench_files:
        print(f"No files matching pattern '*_rmbench_results.json' found in {folder_path}")
        return None
    
    all_results = {}
    summary_stats = []
    
    cnt = 5
    # Process each file
    for rmbench_file in sorted(rmbench_files):
        filename = os.path.basename(rmbench_file)
        step_id = filename.replace('_rmbench_results.json', '')
        results = evaluate_factchecking(rmbench_file, chat_filtered_file)
        if cnt > 0:
            print_evaluation_results(results)
            cnt -= 1
        if results:
            all_results[step_id] = results
            summary_stats.append({
                'step_id': step_id,
                'filename': filename,
                'total_valid_instances': results['total_valid_instances'],
                'hits': results['hits'],
                'hit_rate': results['hit_rate'],
                'hit_rate_percentage': results['hit_rate_percentage']
            })
    
    return {
        'summary_stats': summary_stats,
        'detailed_results': all_results
    }

def print_folder_results(folder_results):
    """
    Print simple step ID and hit rate results, sorted by step ID
    """
    if not folder_results or 'summary_stats' not in folder_results:
        print("No results to display")
        return
    
    summary_stats = folder_results['summary_stats']
    
    # Sort by step_id - try numeric sorting first, fall back to alphabetic
    try:
        # Try to extract numeric part for sorting (e.g., "step1" -> 1)
        def extract_numeric_id(stat):
            step_id = stat['step_id']
            # Extract numbers from the step_id
            numbers = re.findall(r'\d+', step_id)
            if numbers:
                return int(numbers[0])  # Use first number found
            return float('inf')  # Put non-numeric at the end
        
        summary_stats_sorted = sorted(summary_stats, key=extract_numeric_id)
    except:
        # Fall back to alphabetic sorting
        summary_stats_sorted = sorted(summary_stats, key=lambda x: x['step_id'])
    
    print("\nStep ID\t\tHit Rate")
    print("-" * 25)
    
    for stat in summary_stats_sorted:
        print(f"{stat['step_id']}\t\t{stat['hit_rate_percentage']:.2f}%")

def main():
    """
    Main function to run the evaluation on folder
    """
    # Configuration - update these paths as needed
    folder_path = '/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/results/grpo_hs3_16K_step240_clip_max_0.28_qwen3_14b_lr_2e-6_temp_1_kl_0.001_grpo_bs_256_rollout_64_num_prompts_128_r0_fact_base/outputs'  # Current directory, change this to your folder path
    chat_filtered_file = '/lustre/fsw/portfolios/llmservice/users/yizhuj/datasets/rmbench/chat_filtered.json'
    
    # Check if chat_filtered file exists
    if not os.path.exists(chat_filtered_file):
        print(f"Error: {chat_filtered_file} not found")
        return
    
    # Process folder
    folder_results = process_folder(folder_path, chat_filtered_file)
    
    if folder_results:
        print_folder_results(folder_results)

if __name__ == "__main__":
    main()