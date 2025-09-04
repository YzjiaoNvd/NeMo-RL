import json
import os
import glob
import re
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from tqdm import tqdm 
import random

def setup_llm_client(api_key):
    """
    Setup OpenAI client for NVIDIA API
    
    Args:
        api_key (str): API key for NVIDIA
    
    Returns:
        OpenAI: Configured client
    """
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

def llm_judge_evaluation(error_message, factcheck_response, client, model="nvdev/nvidia/llama-3.3-nemotron-super-49b-v1"):
    """
    Use LLM as judge to evaluate if factcheck response correctly identifies the error
    
    Args:
        error_key (str): The error that should be identified
        factcheck_response (str): The factcheck response to evaluate
        client (OpenAI): OpenAI client
        model (str): Model to use for evaluation
    
    Returns:
        dict: Contains 'hit' (bool) and 'reasoning' (str)
    """
    
    prompt = f"""Given an groundtruth error and a fact-checking result, please evaluate whether the factcheck result successfully identifies the groundtruth error. Please note that the fact-checking result is analyzing two responses. Consider the fact-checking result a HIT if any response in it catches the error or closely related concepts. Please strictly follow the output format and make your response as brief as possible. 

### INPUT FORMAT ###
Groundtruth: {error_message}
FACTCHECK RESPONSE: {factcheck_response}

### OUTPUT FORMAT ###
REASONING: [Brief explanation of your decision]
VERDICT: [HIT or MISS]
"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                #{"role": "system", "content": "detailed thinking on"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,  # Low temperature for consistent evaluation
            top_p=0.9,
            max_tokens=5000,
        )
        
        response_text = completion.choices[0].message.content
        
        # Parse the response
        verdict_match = re.search(r'VERDICT:\s*(HIT|MISS)', response_text, re.IGNORECASE)
        reasoning_match = re.search(r'REASONING:\s*(.+)', response_text, re.DOTALL)
        
        if verdict_match:
            verdict = verdict_match.group(1).upper()
            is_hit = verdict == "HIT"
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        else:
            # Fallback parsing
            is_hit = "hit" in response_text.lower() and "miss" not in response_text.lower()
            reasoning = response_text.strip()
        '''
        print(prompt)
        print(is_hit)
        print(reasoning)
        print()
        print()
        '''
        return {
            'hit': is_hit,
            'reasoning': reasoning,
            'raw_response': response_text
        }
        
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return {
            'hit': False,
            'reasoning': f"Error occurred: {str(e)}",
            'raw_response': ""
        }

def evaluate_factchecking(rmbench_file, chat_filtered_file, mode="exact", llm_client=None, rate_limit_delay=1.0, delete_correct_lines=False):
    """
    Evaluate factchecking performance by comparing error_keys from chat_filtered.json
    with factcheck_response content in rmbench_results.json
    
    Args:
        rmbench_file (str): Path to rmbench_results.json
        chat_filtered_file (str): Path to chat_filtered.json
        mode (str): "exact" for exact word matching or "llm" for LLM-as-judge
        llm_client (OpenAI): OpenAI client for LLM mode
        rate_limit_delay (float): Delay between LLM API calls to avoid rate limits
    
    Returns:
        dict: Dictionary containing evaluation results
    """
    
    if mode == "llm" and llm_client is None:
        raise ValueError("LLM client is required for LLM mode")
    
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
    if mode == "llm":
        rmbench_data = rmbench_data[:1000]
    for instance_a in tqdm(rmbench_data):
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
        
        # Get factcheck_response and error_key
        factcheck_response = instance_a.get('factcheck_response')
        error_key = instance_b.get('error_key')
        error_message = instance_b.get('error')
        
        # Evaluate based on mode
        if mode == "exact":
            # Original exact matching logic
            is_hit = False
            reasoning = ""
            match_mode = "all_keywords" # "all_keywords" or "one_keyword"
            if error_key:
                factcheck_lower = factcheck_response.lower()
                if delete_correct_lines:
                    lines = factcheck_lower.strip().split('\n')
                    updated_lines = []
                    for line in lines:
                        if "wrong" in line or "unknown" in line:
                            updated_lines.append(line)
                    factcheck_lower = "\n".join(updated_lines)

                if match_mode == "all_keywords":
                    is_hit = error_key.lower() in factcheck_lower
                else:
                    error_words = re.findall(r'\b\w+\b', error_key.lower())
                    is_hit = any(word in factcheck_lower for word in error_words)


                reasoning = f"Exact match: {is_hit}"
        
        elif mode == "llm":
            # LLM-as-judge evaluation
            if error_message and factcheck_response:
                eval_result = llm_judge_evaluation(error_message, factcheck_response, llm_client)
                is_hit = eval_result['hit']
                reasoning = eval_result['reasoning']
                
                # Rate limiting
                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay)
            else:
                is_hit = False
                reasoning = "Missing error_message or factcheck_response"
        
        if is_hit:
            hits += 1
            
        # Store detailed results for analysis
        detailed_results.append({
            'sample_id': sample_id,
            'chat_id': chat_id,
            'error_key': error_key,
            'error_message': error_message,
            'hit': is_hit,
            'reasoning': reasoning,
            'factcheck_response_snippet': factcheck_response[:500] + '...' if len(factcheck_response) > 500 else factcheck_response
        })
        
    
    # Calculate hit rate
    hit_rate = hits / total_valid_instances if total_valid_instances > 0 else 0
    
    results = {
        'evaluation_mode': mode,
        'total_valid_instances': total_valid_instances,
        'hits': hits,
        'hit_rate': hit_rate,
        'hit_rate_percentage': hit_rate * 100,
        'detailed_results': detailed_results
    }
    
    return results

def process_single_file(args):
    """
    Worker function to process a single rmbench file
    
    Args:
        args (tuple): (rmbench_file, chat_filtered_file, mode, api_keys, rate_limit_delay, worker_id)
    
    Returns:
        tuple: (step_id, filename, results)
    """
    rmbench_file, chat_filtered_file, mode, api_keys, rate_limit_delay, worker_id = args
    
    filename = os.path.basename(rmbench_file)
    step_id = filename.replace('_rmbench_results.json', '')
    
    # Setup LLM client if needed (each process gets its own API key)
    llm_client = None
    if mode == "llm" and api_keys:
        # Use round-robin to assign API key based on worker_id
        api_key = api_keys[worker_id % len(api_keys)]
        llm_client = setup_llm_client(api_key)
        print(f"Worker {worker_id} using API key ending in ...{api_key[-8:]}")
    
    delete_correct_lines = "fact1" in rmbench_file

    print(f"Processing {filename}...")
    results = evaluate_factchecking(rmbench_file, chat_filtered_file, mode, llm_client, rate_limit_delay, delete_correct_lines=delete_correct_lines)
    
    return step_id, filename, results

def print_evaluation_results(results, show_reasoning=False):
    """
    Print formatted evaluation results
    """
    if results is None:
        return
        
    print("=" * 50)
    print(f"FACTCHECKING EVALUATION RESULTS ({results['evaluation_mode'].upper()} MODE)")
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
        if show_reasoning and 'reasoning' in detail:
            print(f"  Reasoning: {detail['reasoning']}")
        print(f"  Factcheck snippet: {detail['factcheck_response_snippet']}")
        print()

def process_folder(folder_path, chat_filtered_file, mode="exact", api_keys=None, rate_limit_delay=0.1, num_processes=None):
    """
    Process all step_id_rmbench_results.json files in a folder using multiprocessing
    
    Args:
        folder_path (str): Path to folder containing the rmbench files
        chat_filtered_file (str): Path to chat_filtered.json
        mode (str): "exact" for exact word matching or "llm" for LLM-as-judge
        api_keys (list): List of API keys for NVIDIA (required for LLM mode)
        rate_limit_delay (float): Delay between LLM API calls
        num_processes (int): Number of processes to use (None for CPU count)
    
    Returns:
        dict: Dictionary containing results for all files
    """
    
    # Check if chat_filtered file exists
    if not os.path.exists(chat_filtered_file):
        print(f"Error: {chat_filtered_file} not found")
        return None
    
    # Validate API keys for LLM mode
    if mode == "llm" and not api_keys:
        print("Error: API keys are required for LLM mode")
        return None
    
    # Find all files matching the pattern
    pattern = os.path.join(folder_path, "*_rmbench_results.json")
    rmbench_files = glob.glob(pattern)
    
    if not rmbench_files:
        print(f"No files matching pattern '*_rmbench_results.json' found in {folder_path}")
        return None
    
    # Determine number of processes - optimize based on number of API keys
    if num_processes is None:
        if mode == "llm" and api_keys:
            # Use multiples of API keys for better distribution
            num_processes = min(len(api_keys) * 2, mp.cpu_count(), len(rmbench_files))
        else:
            num_processes = min(mp.cpu_count(), len(rmbench_files))
    
    print(f"Processing {len(rmbench_files)} files using {num_processes} processes...")
    if mode == "llm":
        print(f"Using {len(api_keys)} API keys with {rate_limit_delay}s delay")
    
    # Prepare arguments for worker processes
    worker_args = []
    for i, rmbench_file in enumerate(rmbench_files):
        worker_args.append((
            rmbench_file, 
            chat_filtered_file, 
            mode, 
            api_keys, 
            rate_limit_delay,
            i  # worker_id for API key selection
        ))
    
    all_results = {}
    summary_stats = []
    
    # Use multiprocessing to process files in parallel
    try:
        with mp.Pool(processes=num_processes) as pool:
            # Use tqdm for progress tracking
            results_list = list(tqdm(
                pool.imap(process_single_file, worker_args),
                total=len(worker_args),
                desc="Processing files"
            ))
        
        # Collect results
        for step_id, filename, results in results_list:
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
    
    except Exception as e:
        print(f"Error in multiprocessing: {e}")
        return None
    
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
    Main function to run the evaluation on folder with multiprocessing
    """
    # Configuration - update these paths as needed
    folder_path = '/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/NeMo-RL/results/grpo_hs3_16K_step240_clip_max_0.28_qwen3_14b_lr_2e-7_temp_1_kl_0.001_grpo_bs_256_rollout_64_num_prompts_128_r0_fact_base/outputs'
    chat_filtered_file = '/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/datasets/fact_checking/rmbench_chat_filtered.json'
    
    # Multiple API keys for faster processing
    API_KEYS = [
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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",
        "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"
    ]
    
    # Configuration for evaluation mode
    EVALUATION_MODE = "llm"  # llm or exact
    RATE_LIMIT_DELAY = 0.00  # Reduced delay since we have multiple API keys
    NUM_PROCESSES = None  # None to use optimized number based on API keys
    
    # Check if chat_filtered file exists
    if not os.path.exists(chat_filtered_file):
        print(f"Error: {chat_filtered_file} not found")
        return
    
    # Process folder with multiprocessing
    if EVALUATION_MODE == "llm":
        print("Using LLM-as-judge mode with multiple API keys")
        folder_results = process_folder(
            folder_path, chat_filtered_file, 
            mode="llm", api_keys=API_KEYS, 
            rate_limit_delay=RATE_LIMIT_DELAY,
            num_processes=NUM_PROCESSES
        )
    else:
        print("Using exact matching mode with multiprocessing")
        folder_results = process_folder(
            folder_path, chat_filtered_file, 
            mode="exact",
            num_processes=NUM_PROCESSES
        )
    
    if folder_results:
        print_folder_results(folder_results)

if __name__ == "__main__":
    main()