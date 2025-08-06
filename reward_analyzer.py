import re
import statistics
from collections import Counter

def parse_reward_file(filename):
    """
    Parse a log file and extract reward calculation data.
    
    Args:
        filename (str): Path to the log file
        
    Returns:
        dict: Dictionary containing lists of reward values
    """
    rewards = {
        'enhanced': [],
        'initial': [],
        'bonus': [],
        'final': []
    }
    
    valid_lines = 0
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                # Check if line contains [REWARD CALC] pattern
                if '[REWARD CALC]' in line:
                    # Extract reward values using regex
                    pattern = r'Enhanced:\s*([-\d.]+),\s*Initial:\s*([-\d.]+),\s*Bonus:\s*([-\d.]+),\s*Final:\s*([-\d.]+)'
                    match = re.search(pattern, line)
                    
                    if match:
                        try:
                            enhanced = float(match.group(1))
                            initial = float(match.group(2))
                            bonus = float(match.group(3))
                            final = float(match.group(4))
                            
                            rewards['enhanced'].append(enhanced)
                            rewards['initial'].append(initial)
                            rewards['bonus'].append(bonus)
                            rewards['final'].append(final)
                            valid_lines += 1
                            
                        except ValueError as e:
                            print(f"Warning: Could not parse values on line {line_num}: {e}")
                    else:
                        print(f"Warning: Line {line_num} contains [REWARD CALC] but doesn't match expected format")
                        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, 0
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, 0
    
    return rewards, valid_lines

def calculate_statistics(values):
    """
    Calculate distribution statistics for a list of values.
    
    Args:
        values (list): List of numerical values
        
    Returns:
        dict: Dictionary containing statistical measures
    """
    if not values:
        return None
    
    # Calculate value frequencies
    value_counts = Counter(values)
    unique_values = len(value_counts)
    
    return {
        'count': len(values),
        'unique_values': unique_values,
        'min': min(values),
        'max': max(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
        'value_counts': dict(value_counts),
        'values': values
    }

def print_distribution_analysis(rewards, valid_lines):
    """
    Print formatted distribution analysis results.
    
    Args:
        rewards (dict): Dictionary containing reward value lists
        valid_lines (int): Number of valid lines processed
    """
    print(f"Processed {valid_lines} valid reward calculation lines\n")
    print("REWARD DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    metric_names = {
        'enhanced': 'Enhanced Reward',
        'initial': 'Initial Reward', 
        'bonus': 'Bonus from Fact Checking',
        'final': 'Final Reward'
    }
    
    for metric, name in metric_names.items():
        stats = calculate_statistics(rewards[metric])
        
        if stats:
            print(f"\n{name}:")
            print(f"  Total Count: {stats['count']}")
            print(f"  Unique Values: {stats['unique_values']}")
            print(f"  Min: {stats['min']}")
            print(f"  Max: {stats['max']}")
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Median: {stats['median']}")
            print(f"  Std Dev: {stats['std_dev']:.3f}")
            
            # Print value frequencies
            print(f"  Value Frequencies:")
            sorted_counts = sorted(stats['value_counts'].items())
            for value, count in sorted_counts:
                percentage = (count / stats['count']) * 100
                print(f"    {value}: {count} times ({percentage:.1f}%)")
                
        else:
            print(f"\n{name}: No data found")

def main():
    """
    Main function to run the reward distribution analysis.
    """
    # Specify your file name here
    filename = "/lustre/fsw/portfolios/llmservice/users/yizhuj/NeMo-RL/4434739-logs/ray-driver.log"  # Change this to your actual file name
    
    # Parse the file
    rewards, valid_lines = parse_reward_file(filename)
    
    if rewards is not None and valid_lines > 0:
        # Print analysis
        print_distribution_analysis(rewards, valid_lines)
        
        # Optional: Save results to a summary file
        save_summary = input("\nSave summary to file? (y/n): ").lower().strip()
        if save_summary == 'y':
            summary_filename = f"{filename}_summary.txt"
            with open(summary_filename, 'w') as f:
                f.write(f"Reward Distribution Analysis for {filename}\n")
                f.write("=" * 50 + "\n\n")
                
                metric_names = {
                    'enhanced': 'Enhanced Reward',
                    'initial': 'Initial Reward', 
                    'bonus': 'Bonus from Fact Checking',
                    'final': 'Final Reward'
                }
                
                for metric, name in metric_names.items():
                    stats = calculate_statistics(rewards[metric])
                    if stats:
                        f.write(f"{name}:\n")
                        f.write(f"  Total Count: {stats['count']}\n")
                        f.write(f"  Unique Values: {stats['unique_values']}\n")
                        f.write(f"  Min: {stats['min']}\n")
                        f.write(f"  Max: {stats['max']}\n")
                        f.write(f"  Mean: {stats['mean']:.3f}\n")
                        f.write(f"  Median: {stats['median']}\n")
                        f.write(f"  Std Dev: {stats['std_dev']:.3f}\n")
                        
                        # Write value frequencies
                        f.write(f"  Value Frequencies:\n")
                        sorted_counts = sorted(stats['value_counts'].items())
                        for value, count in sorted_counts:
                            percentage = (count / stats['count']) * 100
                            f.write(f"    {value}: {count} times ({percentage:.1f}%)\n")
                        f.write("\n")
            
            print(f"Summary saved to {summary_filename}")
    else:
        print("No valid reward calculation lines found in the file.")

if __name__ == "__main__":
    main()