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
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from queue import Queue

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

def llm_judge_evaluation(error_message, factcheck_response, api_key, model="nvdev/nvidia/llama-3.3-nemotron-super-49b-v1"):
    """
    Use LLM as judge to evaluate if factcheck response correctly identifies the error
    
    Args:
        error_message (str): The groundtruth errors (formatted factcheck)
        factcheck_response (str): The factcheck response to evaluate
        api_key (str): API key for NVIDIA
        model (str): Model to use for evaluation
    
    Returns:
        dict: Contains error counts and reasoning
    """
    
    # Create client inside the function for multiprocessing compatibility
    client = setup_llm_client(api_key)
    
    prompt = f"""Count the errors in the groundtruth (only related to "II") and how many are caught in the factcheck response. Please strictly follow the output format requirement and make your reply brief.

GROUNDTRUTH:
{error_message}

FACTCHECK RESPONSE:
{factcheck_response}

Instructions:
1. Count total errors in groundtruth for Response 1
2. Count total errors in groundtruth for Response 2  
3. Count how many Response 1 errors are caught in factcheck
4. Count how many Response 2 errors are caught in factcheck

OUTPUT FORMAT:
RESPONSE 1 ERRORS: [pure number]
RESPONSE 1 CAUGHT: [pure number]
RESPONSE 2 ERRORS: [pure number] 
RESPONSE 2 CAUGHT: [pure number]
TOTAL ERRORS: [pure number]
TOTAL CAUGHT: [pure number]
"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Low temperature for consistent counting
            top_p=0.9,
            max_tokens=500,
        )
        
        response_text = completion.choices[0].message.content

        # Parse the response
        r1_errors = re.search(r'RESPONSE 1 ERRORS:\s*(\d+)', response_text)
        r1_caught = re.search(r'RESPONSE 1 CAUGHT:\s*(\d+)', response_text)
        r2_errors = re.search(r'RESPONSE 2 ERRORS:\s*(\d+)', response_text)
        r2_caught = re.search(r'RESPONSE 2 CAUGHT:\s*(\d+)', response_text)
        total_errors = re.search(r'TOTAL ERRORS:\s*(\d+)', response_text)
        total_caught = re.search(r'TOTAL CAUGHT:\s*(\d+)', response_text)
        
        # Extract numbers
        r1_errors_count = int(r1_errors.group(1)) if r1_errors else 0
        r1_caught_count = int(r1_caught.group(1)) if r1_caught else 0
        r2_errors_count = int(r2_errors.group(1)) if r2_errors else 0
        r2_caught_count = int(r2_caught.group(1)) if r2_caught else 0
        total_errors_count = int(total_errors.group(1)) if total_errors else 0
        total_caught_count = int(total_caught.group(1)) if total_caught else 0
        
        # Calculate hit rate
        hit_rate = total_caught_count / total_errors_count if total_errors_count > 0 else 0
        is_hit = hit_rate == 1.0  # Consider it a hit if all errors are caught

        return {
            'hit': is_hit,
            'reasoning': f"Response 1: {r1_caught_count}/{r1_errors_count} errors caught. Response 2: {r2_caught_count}/{r2_errors_count} errors caught. Total: {total_caught_count}/{total_errors_count} ({hit_rate:.1%})",
            'raw_response': response_text,
            'r1_errors': r1_errors_count,
            'r1_caught': r1_caught_count,
            'r2_errors': r2_errors_count,
            'r2_caught': r2_caught_count,
            'total_errors': total_errors_count,
            'total_caught': total_caught_count,
            'hit_rate': hit_rate
        }
        
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return {
            'hit': False,
            'reasoning': f"Error occurred: {str(e)}",
            'raw_response': "",
            'r1_errors': 0,
            'r1_caught': 0,
            'r2_errors': 0,
            'r2_caught': 0,
            'total_errors': 0,
            'total_caught': 0,
            'hit_rate': 0.0
        }

def load_json_file(file_path):
    """Load JSON file and return data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def find_step_files(folder_path, pattern="*hs3local_results.json"):
    """
    Find and sort step files in the given folder
    
    Args:
        folder_path (str): Path to folder containing step files
        pattern (str): File pattern to match
        
    Returns:
        list: Sorted list of tuples (step_number, file_path)
    """
    folder_path = Path(folder_path)
    step_files = []
    
    # Find all matching files
    for file_path in folder_path.glob(pattern):
        filename = file_path.name
        # Extract step number from filename like "step_5_hs3local_results.json"
        step_match = re.search(r'step_(\d+)', filename)
        if step_match:
            step_num = int(step_match.group(1))
            step_files.append((step_num, str(file_path)))
        else:
            print(f"Warning: Could not extract step number from {filename}")
    
    # Sort by step number
    step_files.sort(key=lambda x: x[0])
    
    print(f"Found {len(step_files)} step files:")
    for step_num, file_path in step_files:
        print(f"  Step {step_num}: {Path(file_path).name}")
    
    return step_files

def match_instances_by_context(file1_data, file2_data):
    """
    Match instances between files based on context value
    
    Args:
        file1_data (list): Data from file 1 (helpfulness reasoning)
        file2_data (list): Data from file 2 (factcheck responses)
    
    Returns:
        list: Matched instances with combined data
    """
    matches = []
    
    # Create lookup dict for file2 data by context
    file2_lookup = {}
    for item in file2_data:
        # Context might be in metadata or directly in the item
        if 'metadata' in item and 'context' in item['metadata']:
            context = item['metadata']['context']
        elif 'context' in item:
            context = item['context']
        else:
            continue
        
        file2_lookup[context] = item
    
    # Match file1 instances with file2
    for item1 in file1_data:
        context = item1.get('context', '')
        if context in file2_lookup:
            matched_item = {
                'file1_data': item1,
                'file2_data': file2_lookup[context],
                'context_preview': context[:100] + "..." if len(context) > 100 else context
            }
            matches.append(matched_item)
    
    return matches

def format_helpfulness_as_factcheck(helpfulness1, helpfulness2):
    """
    Format helpfulness reasoning into factcheck format
    
    Args:
        helpfulness1 (str): Helpfulness reasoning for response 1
        helpfulness2 (str): Helpfulness reasoning for response 2
        
    Returns:
        str: Formatted factcheck groundtruth
    """
    formatted = "[Fact Checking for Response 1]\n"
    if helpfulness1:
        formatted += helpfulness1.strip()
    formatted += "\n[End of Fact Checking for Response 1]\n\n"
    
    formatted += "[Fact Checking for Response 2]\n"
    if helpfulness2:
        formatted += helpfulness2.strip()
    formatted += "\n[End of Fact Checking for Response 2]"
    
    return formatted

def extract_factcheck_content(factcheck_response):
    """
    Extract content between [Fact Checking for Response 1] and [End of Fact Checking for Response 2]
    
    Args:
        factcheck_response (str): Full factcheck response
        
    Returns:
        str: Extracted factcheck content only
    """
    # Find start marker
    start_pattern = r'\[Fact Checking for Response 1\]'
    end_pattern = r'\[End of Fact Checking for Response 2\]'
    
    start_match = re.search(start_pattern, factcheck_response)
    end_match = re.search(end_pattern, factcheck_response)
    
    if start_match and end_match:
        start_pos = start_match.start()
        end_pos = end_match.end()
        extracted = factcheck_response[start_pos:end_pos]
        return extracted
    
    # Fallback: if markers not found, return the original response
    return factcheck_response

def handle_response_switching(match):
    """
    Handle cases where response1 and response2 might be switched
    by comparing actual response content between files to find correct alignment
    
    Args:
        match (dict): Matched instance
        
    Returns:
        bool: True if responses are switched
    """
    # Get responses from both files
    file1_response1 = match['file1_data'].get('response1', '')
    file1_response2 = match['file1_data'].get('response2', '')
    
    # Get responses from file2 (could be in metadata or direct)
    file2_response1 = ''
    file2_response2 = ''
    
    if 'metadata' in match['file2_data']:
        file2_response1 = match['file2_data']['metadata'].get('response1', '')
        file2_response2 = match['file2_data']['metadata'].get('response2', '')
    else:
        file2_response1 = match['file2_data'].get('response1', '')
        file2_response2 = match['file2_data'].get('response2', '')
    
    # Compare responses to determine correct alignment
    response1_matches_1 = file1_response1 == file2_response1
    response1_matches_2 = file1_response1 == file2_response2
    response2_matches_1 = file1_response2 == file2_response1
    response2_matches_2 = file1_response2 == file2_response2
    
    # Determine if responses are switched
    responses_switched = response1_matches_2 and response2_matches_1
    return responses_switched

def worker_evaluate_match(args):
    """
    Worker function for multiprocessing - evaluates a single match
    
    Args:
        args (tuple): (match, api_key, worker_id)
        
    Returns:
        dict: Evaluation results
    """
    match, api_key, worker_id = args
    question_id = match['file1_data'].get('question_id', 'unknown')
    
    try:
        # Handle potential response switching and get correct alignment
        responses_switched = handle_response_switching(match)
        
        # Format helpfulness reasoning into factcheck format (groundtruth)
        helpfulness1 = match['file1_data']['helpfulness_reasoning1']
        helpfulness2 = match['file1_data']['helpfulness_reasoning2']
        if responses_switched:
            formatted_groundtruth = format_helpfulness_as_factcheck(helpfulness2, helpfulness1)
        else:
            formatted_groundtruth = format_helpfulness_as_factcheck(helpfulness1, helpfulness2)
        
        # Get actual factcheck response and extract only the factcheck content
        full_factcheck_response = match['file2_data']['factcheck_response']
        actual_factcheck = extract_factcheck_content(full_factcheck_response)
        
        # LLM evaluation: does factcheck identify all issues in groundtruth?
        evaluation = llm_judge_evaluation(formatted_groundtruth, actual_factcheck, api_key)
        
        results = {
            'question_id': question_id,
            'context_preview': match['context_preview'],
            'formatted_groundtruth': formatted_groundtruth,
            'actual_factcheck': actual_factcheck,
            'evaluation': evaluation,
            'worker_id': worker_id
        }
        
        return results
        
    except Exception as e:
        print(f"Error evaluating {question_id} in worker {worker_id}: {e}")
        return {
            'question_id': question_id,
            'context_preview': match['context_preview'],
            'error': str(e),
            'evaluation': {'hit': False, 'reasoning': f"Error: {str(e)}", 'raw_response': ""},
            'worker_id': worker_id
        }

def evaluate_all_matches_multiprocess(matches, api_key, max_workers=None):
    """
    Evaluate all matched instances using multiprocessing
    
    Args:
        matches (list): List of matched instances
        api_key (str): API key for NVIDIA
        max_workers (int): Number of parallel workers (None for CPU count)
        
    Returns:
        list: List of evaluation results
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(matches))
    
    # Prepare arguments for worker processes
    worker_args = [(match, api_key, i % max_workers) for i, match in enumerate(matches)]
    
    results = []
    completed_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(worker_evaluate_match, args): args for args in worker_args}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(matches), desc="Evaluating matches") as pbar:
            for future in as_completed(future_to_args):
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error processing task: {e}")
                    pbar.update(1)
    
    return results

def evaluate_all_matches_threading(matches, api_key, max_workers=8):
    """
    Alternative implementation using threading for I/O bound tasks
    
    Args:
        matches (list): List of matched instances
        api_key (str): API key for NVIDIA
        max_workers (int): Number of threads
        
    Returns:
        list: List of evaluation results
    """
    results = []
    results_lock = threading.Lock()
    
    def thread_worker():
        while True:
            try:
                match, index = work_queue.get_nowait()
            except:
                break
                
            try:
                worker_args = (match, api_key, threading.current_thread().ident)
                result = worker_evaluate_match(worker_args)
                result['original_index'] = index
                
                with results_lock:
                    results.append(result)
                
            except Exception as e:
                print(f"Error in thread worker: {e}")
            finally:
                work_queue.task_done()
    
    # Create work queue
    work_queue = Queue()
    for i, match in enumerate(matches):
        work_queue.put((match, i))
    
    # Start threads
    threads = []
    for _ in range(max_workers):
        t = threading.Thread(target=thread_worker)
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Wait for completion with progress bar
    with tqdm(total=len(matches), desc="Evaluating matches") as pbar:
        last_count = 0
        while len(results) < len(matches):
            time.sleep(1)
            current_count = len(results)
            pbar.update(current_count - last_count)
            last_count = current_count
    
    # Wait for all threads to finish
    for t in threads:
        t.join()
    
    # Sort results by original order
    results.sort(key=lambda x: x.get('original_index', 0))
    return results

def generate_summary_report(evaluation_results):
    """
    Generate a summary report of the evaluation results
    
    Args:
        evaluation_results (list): List of evaluation results
        
    Returns:
        dict: Summary statistics
    """
    total_instances = len(evaluation_results)
    total_hits = 0
    total_errors = 0
    total_caught = 0
    r1_total_errors = 0
    r1_total_caught = 0
    r2_total_errors = 0
    r2_total_caught = 0
    individual_hit_rates = []
    
    for result in evaluation_results:
        if 'evaluation' in result:
            eval_data = result['evaluation']
            if eval_data.get('hit', False):
                total_hits += 1
            
            # Collect individual hit rates
            individual_hit_rates.append(eval_data.get('hit_rate', 0.0))
            # Sum up error counts
            total_errors += eval_data.get('total_errors', 0)
            total_caught += eval_data.get('total_caught', 0)
            r1_total_errors += eval_data.get('r1_errors', 0)
            r1_total_caught += eval_data.get('r1_caught', 0)
            r2_total_errors += eval_data.get('r2_errors', 0)
            r2_total_caught += eval_data.get('r2_caught', 0)

    # Calculate overall statistics
    overall_hit_rate = total_hits / total_instances if total_instances > 0 else 0
    overall_error_catch_rate = total_caught / total_errors if total_errors > 0 else 0
    average_hit_rate = sum(individual_hit_rates) / len(individual_hit_rates) if individual_hit_rates else 0
    r1_catch_rate = r1_total_caught / r1_total_errors if r1_total_errors > 0 else 0
    r2_catch_rate = r2_total_caught / r2_total_errors if r2_total_errors > 0 else 0
    
    summary = {
        'total_instances': total_instances,
        'total_hits': total_hits,
        'overall_hit_rate': overall_hit_rate,
        'average_hit_rate': average_hit_rate,
        'error_analysis': {
            'total_errors': total_errors,
            'total_caught': total_caught,
            'overall_catch_rate': overall_error_catch_rate,
            'response1': {
                'errors': r1_total_errors,
                'caught': r1_total_caught,
                'catch_rate': r1_catch_rate
            },
            'response2': {
                'errors': r2_total_errors,
                'caught': r2_total_caught,
                'catch_rate': r2_catch_rate
            }
        }
    }
    
    return summary

def evaluate_single_step(file1_data, step_num, step_file_path, api_key, max_workers, use_threading, limit_instances=None):
    """
    Evaluate a single step file
    
    Args:
        file1_data (list): Helpfulness reasoning data
        step_num (int): Step number
        step_file_path (str): Path to step file
        api_key (str): API key
        max_workers (int): Number of workers
        use_threading (bool): Whether to use threading
        limit_instances (int): Limit number of instances for testing
        
    Returns:
        dict: Step evaluation results
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING STEP {step_num}")
    print(f"File: {Path(step_file_path).name}")
    print(f"{'='*60}")
    
    # Load step file data
    file2_data = load_json_file(step_file_path)
    if not file2_data:
        print(f"Error: Could not load step file {step_file_path}")
        return None
    
    print(f"Loaded {len(file2_data)} instances from step file")
    
    # Match instances by context
    matches = match_instances_by_context(file1_data, file2_data)
    
    if not matches:
        print(f"Error: No matches found for step {step_num}")
        return None
    
    # Limit instances if specified (for testing)
    if limit_instances:
        matches = matches[:limit_instances]
        print(f"Limited to {len(matches)} instances for testing")
    
    print(f"Found {len(matches)} matches for step {step_num}")
    
    # Evaluate matches
    start_time = time.time()
    
    if use_threading:
        evaluation_results = evaluate_all_matches_threading(matches, api_key, max_workers or 8)
    else:
        evaluation_results = evaluate_all_matches_multiprocess(matches, api_key, max_workers)
    
    end_time = time.time()
    
    # Generate summary
    summary = generate_summary_report(evaluation_results)
    
    step_result = {
        'step_number': step_num,
        'step_file': Path(step_file_path).name,
        'total_instances': len(matches),
        'evaluation_time': end_time - start_time,
        'summary': summary,
        'detailed_results': evaluation_results
    }
    
    print(f"Step {step_num} completed in {end_time - start_time:.2f} seconds")
    print(f"Hit rate: {summary['average_hit_rate']:.2%} ({summary['total_hits']}/{summary['total_instances']})")
    
    return step_result

def generate_cross_step_summary(all_step_results):
    """
    Generate summary across all steps
    
    Args:
        all_step_results (list): Results from all steps
        
    Returns:
        dict: Cross-step summary
    """
    step_summaries = []
    
    for step_result in all_step_results:
        if step_result:
            summary = step_result['summary']
            step_summary = {
                'step_number': step_result['step_number'],
                'step_file': step_result['step_file'],
                'total_instances': summary['total_instances'],
                'total_hits': summary['total_hits'],
                'overall_hit_rate': summary['overall_hit_rate'],
                'average_hit_rate': summary['average_hit_rate'],
                'total_errors': summary['error_analysis']['total_errors'],
                'total_caught': summary['error_analysis']['total_caught'],
                'overall_catch_rate': summary['error_analysis']['overall_catch_rate'],
                'evaluation_time': step_result['evaluation_time']
            }
            step_summaries.append(step_summary)
    
    # Sort by step number
    step_summaries.sort(key=lambda x: x['step_number'])
    
    # Calculate overall statistics
    total_instances_all = sum(s['total_instances'] for s in step_summaries)
    total_hits_all = sum(s['total_hits'] for s in step_summaries)
    total_time_all = sum(s['evaluation_time'] for s in step_summaries)
    
    cross_step_summary = {
        'total_steps_evaluated': len(step_summaries),
        'total_instances_all_steps': total_instances_all,
        'total_hits_all_steps': total_hits_all,
        'overall_hit_rate_all_steps': total_hits_all / total_instances_all if total_instances_all > 0 else 0,
        'total_evaluation_time': total_time_all,
        'step_by_step_results': step_summaries
    }
    
    return cross_step_summary

def main(file1_path, step_folder_path, api_key, output_dir="./factcheck_step_evaluation_results", 
         max_workers=None, use_threading=False, limit_instances=None, file_pattern="*hs3local_results.json"):
    """
    Main function to run evaluation across multiple step files
    
    Args:
        file1_path (str): Path to file 1 (helpfulness reasoning)
        step_folder_path (str): Path to folder containing step files
        api_key (str): API key for NVIDIA
        output_dir (str): Directory to save results
        max_workers (int): Number of parallel workers
        use_threading (bool): Whether to use threading instead of multiprocessing
        limit_instances (int): Limit instances per step for testing
        file_pattern (str): Pattern to match step files
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading helpfulness reasoning file...")
    file1_data = load_json_file(file1_path)
    
    if not file1_data:
        print("Error: Could not load helpfulness reasoning file")
        return
    
    print(f"Loaded {len(file1_data)} instances from helpfulness reasoning file")
    
    # Find all step files
    step_files = find_step_files(step_folder_path, file_pattern)
    
    if not step_files:
        print(f"Error: No step files found in {step_folder_path}")
        return
    
    print(f"\nStarting evaluation of {len(step_files)} step files...")
    
    # Evaluate each step
    all_step_results = []
    
    for step_num, step_file_path in step_files:
        step_result = evaluate_single_step(
            file1_data, step_num, step_file_path, api_key, 
            max_workers, use_threading, limit_instances
        )
        
        if step_result:
            all_step_results.append(step_result)
            
            # Save individual step results
            step_output_path = os.path.join(output_dir, f"step_{step_num}_detailed_results.json")
            with open(step_output_path, 'w', encoding='utf-8') as f:
                json.dump(step_result, f, indent=2, ensure_ascii=False)
    
    # Generate cross-step summary
    print(f"\n{'='*60}")
    print("GENERATING CROSS-STEP SUMMARY")
    print(f"{'='*60}")
    
    cross_step_summary = generate_cross_step_summary(all_step_results)
    
    # Save cross-step summary
    summary_output_path = os.path.join(output_dir, "cross_step_summary.json")
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        json.dump(cross_step_summary, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total steps evaluated: {cross_step_summary['total_steps_evaluated']}")
    print(f"Total instances across all steps: {cross_step_summary['total_instances_all_steps']}")
    print(f"Total evaluation time: {cross_step_summary['total_evaluation_time']:.2f} seconds")
    print(f"\nStep-by-step hit rates:")
    
    for step_summary in cross_step_summary['step_by_step_results']:
        print(f"  Step {step_summary['step_number']:3d}: {step_summary['average_hit_rate']:6.2%} "
              f"({step_summary['total_hits']:3d}/{step_summary['total_instances']:3d}) - "
              f"{step_summary['step_file']}")
    
    print(f"\nOverall hit rate across all steps: {cross_step_summary['overall_hit_rate_all_steps']:.2%}")
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate factcheck coverage across multiple training steps")
    parser.add_argument("--file1", 
                       default='/lustre/fsw/portfolios/llmservice/users/yizhuj/datasets/fact_checking/hs3_fact.json', 
                       help="Path to file 1 (helpfulness reasoning)")
    parser.add_argument("--step_folder", 
                       default='/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/NeMo-RL/results/grpo_hs3_16K_step240_clip_max_0.28_qwen3_14b_lr_2e-8_temp_1_kl_0.001_grpo_bs_256_rollout_64_num_prompts_128_r0_fact1_base/outputs/', 
                       help="Path to folder containing step files")
    parser.add_argument("--api_key", 
                       default="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", 
                       help="NVIDIA API key")
    parser.add_argument("--max_workers", type=int, default=None, 
                       help="Number of parallel workers (defaults to CPU count)")
    parser.add_argument("--use_threading", action="store_true", 
                       help="Use threading instead of multiprocessing")
    parser.add_argument("--limit_instances", type=int, default=None, 
                       help="Limit number of instances per step for testing")
    parser.add_argument("--file_pattern", default="*hs3local_results.json", 
                       help="Pattern to match step files")
    
    args = parser.parse_args()
    
    main(args.file1, args.step_folder, args.api_key, 
         max_workers=args.max_workers, use_threading=args.use_threading, 
         limit_instances=args.limit_instances, file_pattern=args.file_pattern)