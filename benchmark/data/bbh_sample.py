import os
import json
import random
from pathlib import Path

def read_jsonl(file_path):
    """Read all entries from a JSONL file."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries

def sample_entries(entries, sample_size):
    """Randomly sample entries without replacement."""
    return random.sample(entries, min(sample_size, len(entries)))

def main(bbh_dir, output_test_path, samples_per_file=2):
    """
    Create bbh_test.jsonl by sampling from each JSONL file in bbh_dir, adding task_id to each entry.
    
    Args:
        bbh_dir (str): Path to the BBH directory containing JSONL files.
        output_test_path (str): Path to save bbh_test.jsonl.
        samples_per_file (int): Number of samples per file for the test dataset.
    """
    # Ensure output directory exists
    Path(output_test_path).parent.mkdir(parents=True, exist_ok=True)
    
    test_samples = []
    
    # List all JSONL files in the BBH directory
    jsonl_files = [f for f in os.listdir(bbh_dir) if f.endswith('.jsonl')]
    
    for jsonl_file in jsonl_files:
        file_path = os.path.join(bbh_dir, jsonl_file)
        entries = read_jsonl(file_path)
        
        # Get task name from file name (without .jsonl)
        task_name = jsonl_file.replace('.jsonl', '')
        
        # Sample entries
        sampled = sample_entries(entries, samples_per_file)
        
        # Add task_id to each sampled entry
        for idx, entry in enumerate(sampled):
            entry['task_id'] = f"{task_name}_{idx}"
            test_samples.append(entry)
    
    # Write test samples
    with open(output_test_path, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Created {output_test_path} with {len(test_samples)} entries.")

if __name__ == "__main__":
    bbh_directory = "./bbh" 
    test_output = "./bbh_test.jsonl"
    main(bbh_directory, test_output)