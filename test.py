#!/usr/bin/env python3
"""Test script to diagnose dataset loading issues."""

import json
from datasets import load_dataset, Dataset

def test_dataset_loading():
    """Test loading various reward benchmark datasets."""
    
    print("="*60)
    print("Testing Dataset Loading")
    print("="*60)
    
    # Test configurations for different datasets
    test_configs = [
        # JudgeBench variations
        ("ScalerLab/JudgeBench", None, ["gpt"]),
        
    ]
    
    results = []
    
    for dataset_name, config, splits in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {dataset_name}")
        if config:
            print(f"Config: {config}")
        print(f"{'='*60}")
        
        for split in splits:
            try:
                print(f"\n  Trying split '{split}'...")
                
                if config:
                    ds = load_dataset(dataset_name, config, split=split)
                else:
                    ds = load_dataset(dataset_name, split=split)
                
                print(f"  ✓ SUCCESS! Loaded {len(ds)} examples")
                
                # Show dataset info
                if len(ds) > 0:
                    print(f"  Fields: {list(ds[0].keys())}")
                    print(f"  Sample (first 3 fields):")
                    sample = ds[0]
                    for i, (k, v) in enumerate(sample.items()):
                        if i >= 3:
                            break
                        v_str = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                        print(f"    {k}: {v_str}")
                
                results.append({
                    "dataset": dataset_name,
                    "config": config,
                    "split": split,
                    "success": True,
                    "num_examples": len(ds),
                    "fields": list(ds[0].keys()) if len(ds) > 0 else []
                })
                
            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                results.append({
                    "dataset": dataset_name,
                    "config": config,
                    "split": split,
                    "success": False,
                    "error": str(e)
                })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r["success"]]
    if successful:
        print("\n✓ Successfully loaded:")
        for r in successful:
            config_str = f" (config: {r['config']})" if r['config'] else ""
            print(f"  - {r['dataset']}{config_str} [{r['split']}]: {r['num_examples']} examples")
    else:
        print("\n✗ No datasets were successfully loaded!")
    
    # Save results
    with open("dataset_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: dataset_test_results.json")


def create_local_dataset_example():
    """Show how to create a local dataset for testing."""
    
    print(f"\n{'='*60}")
    print("Creating Local Dataset Example")
    print(f"{'='*60}")
    
    # Example data in the expected format
    data = {
        "prompt": [
            "What is machine learning?",
            "Explain the theory of relativity",
            "How do I make a good cup of coffee?",
        ],
        "response_1": [
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions with minimal human intervention.",
            "The theory of relativity, proposed by Albert Einstein, consists of two parts: special relativity (1905) and general relativity (1915). Special relativity states that the laws of physics are the same in all inertial reference frames and that the speed of light is constant. General relativity extends this to include gravity as a curvature of spacetime.",
            "To make a good cup of coffee: 1) Use fresh, quality beans. 2) Grind beans just before brewing. 3) Use the right water temperature (195-205°F). 4) Maintain proper coffee-to-water ratio (1:15 to 1:17). 5) Brew for the appropriate time based on your method.",
        ],
        "response_2": [
            "Machine learning is when computers learn stuff by themselves.",
            "Einstein said everything is relative and E=mc². It's about how time and space work differently at high speeds.",
            "Just put coffee in hot water and add sugar and milk if you want.",
        ],
        "score_1": [5, 5, 4],
        "score_2": [2, 3, 2],
        "preference_label": [1, 2, 2],  # 1-3 means response 1 is better
    }
    
    # Create dataset
    ds = Dataset.from_dict(data)
    
    # Save to disk
    ds.save_to_disk("local_judgebench_dataset")
    
    # Also save as JSON for easy editing
    with open("local_judgebench_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print("✓ Created local dataset with {} examples".format(len(ds)))
    print("  - Saved to: local_judgebench_dataset/")
    print("  - JSON version: local_judgebench_data.json")
    
    # Show how to load it
    print("\nTo use this local dataset, modify JudgeBenchDataset.__init__():")
    print("""
    # Replace the dataset loading with:
    from datasets import load_from_disk
    ds = load_from_disk("local_judgebench_dataset")
    
    # Or load from JSON:
    ds = load_dataset("json", data_files="local_judgebench_data.json", split="train")
    """)


if __name__ == "__main__":
    # Test dataset loading
    test_dataset_loading()
    
    # Create example local dataset
    create_local_dataset_example()
    
    print("\n✅ Testing complete!")