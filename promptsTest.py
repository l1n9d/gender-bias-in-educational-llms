import pandas as pd
import csv

def load_prompts_from_csv(file_path):
    """Load and organize prompts from a CSV file."""
    prompts_dict = {}
    current_category = None
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if this is a category line (category lines don't have quotes)
            if '"""' not in line:
                # Extract category name
                if line.startswith(tuple("0123456789")) and "." in line:
                    # Format like "1. Category Name"
                    parts = line.split('. ', 1)
                    if len(parts) > 1:
                        current_category = parts[1].strip()
                        prompts_dict[current_category] = []
                else:
                    # Possible category header
                    current_category = line
                    prompts_dict[current_category] = []
            
            # Check if this is a prompt line (has quotes)
            elif '"""' in line and current_category:
                # Extract prompt from between quotes
                prompt = line.split('"""')[1].strip()
                if prompt:
                    prompts_dict[current_category].append(prompt)
    
    return prompts_dict

# Usage
prompts = load_prompts_from_csv('prompts.csv')

# Print to verify
for category, prompt_list in prompts.items():
    print(f"\n{category}:")
    for prompt in prompt_list:
        print(f"  - {prompt}")