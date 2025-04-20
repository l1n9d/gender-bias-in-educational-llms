import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

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
                
            # Check if this is a category line
            if '"""' not in line:
                # Extract category name
                if line.startswith(tuple("0123456789")) and "." in line:
                    parts = line.split('. ', 1)
                    if len(parts) > 1:
                        current_category = parts[1].strip()
                        prompts_dict[current_category] = []
                else:
                    current_category = line
                    prompts_dict[current_category] = []
            
            # Check if this is a prompt line
            elif '"""' in line and current_category:
                prompt = line.split('"""')[1].strip()
                if prompt:
                    prompts_dict[current_category].append(prompt)
    
    return prompts_dict

def generate_mock_responses(prompts, model_type="mock"):
    """Generate mock responses with realistic bias patterns."""
    results = []
    
    # Different bias patterns for different models
    if model_type == "mock1":
        # Model 1: Slight male bias
        male_bias = 0.6
        toxicity_range = (0.1, 0.3)
    else:
        # Model 2: More balanced
        male_bias = 0.45
        toxicity_range = (0.05, 0.2)
    
    for prompt in prompts:
        # Simulate different response patterns based on prompt content
        content = f"Mock response for: {prompt[:50]}..."
        
        # Adjust bias based on prompt content
        prompt_lower = prompt.lower()
        local_bias = male_bias
        
        # Adjust bias if prompt mentions specific genders
        if "female" in prompt_lower or "woman" in prompt_lower or "women" in prompt_lower:
            local_bias -= 0.2
        if "male" in prompt_lower or "man" in prompt_lower or "men" in prompt_lower:
            local_bias += 0.1
            
        # Generate counts based on bias
        total_terms = np.random.randint(5, 15)
        male_count = int(total_terms * local_bias)
        female_count = total_terms - male_count
        
        # Calculate metrics
        gender_balance = female_count / total_terms if total_terms > 0 else 0.5
        
        # Generate toxicity based on content
        base_toxicity = np.random.uniform(*toxicity_range)
        toxicity = base_toxicity
        
        # Add additional analysis metrics
        technical_terms = np.random.randint(5, 20)
        stem_references = np.random.randint(3, 12)
        diversity_score = np.random.uniform(0.2, 0.8)
        
        results.append({
            "prompt": prompt,
            "response": content,
            "model": model_type,
            "he": male_count,
            "she": female_count,
            "gender_balance": gender_balance,
            "toxicity": toxicity,
            "technical_terms": technical_terms,
            "stem_references": stem_references,
            "diversity_score": diversity_score,
            "prompt_category": get_prompt_category(prompt, prompts_dict)
        })
    
    return pd.DataFrame(results)

def get_prompt_category(prompt, prompts_dict):
    """Determine which category a prompt belongs to."""
    for category, prompts in prompts_dict.items():
        if prompt in prompts:
            return category
    return "Unknown"

def compare_models(prompts, prompts_dict):
    """Generate and compare model responses."""
    # Use mock data generation
    mock_df1 = generate_mock_responses(prompts, "mock1")
    mock_df2 = generate_mock_responses(prompts, "mock2")
    
    combined_df = pd.concat([mock_df1, mock_df2], ignore_index=True)
    
    return combined_df

def generate_visualizations(df, output_prefix):
    """Generate insightful visualizations from the results."""
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('analysis_results'):
        os.makedirs('analysis_results')
    
    # 1. Overall gender balance comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='gender_balance', data=df)
    plt.title('Gender Balance by Model')
    plt.ylabel('Female Representation (0-1)')
    plt.tight_layout()
    plt.savefig(f'analysis_results/{output_prefix}_gender_balance.png')
    
    # 2. Category-based analysis
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='prompt_category', y='gender_balance', hue='model', data=df)
    plt.title('Gender Balance by Prompt Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'analysis_results/{output_prefix}_category_analysis.png')
    
    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_columns = ['gender_balance', 'toxicity', 'technical_terms', 'stem_references', 'diversity_score']
    corr_matrix = df[corr_columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Between Metrics')
    plt.tight_layout()
    plt.savefig(f'analysis_results/{output_prefix}_correlation.png')
    
    # Return for display in notebook if needed
    return plt

# Execution
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists('analysis_results'):
        os.makedirs('analysis_results')
    # Load prompts from CSV
    prompts_dict = load_prompts_from_csv('prompts.csv')
    
    # Flatten prompts for analysis
    all_prompts = []
    for category_prompts in prompts_dict.values():
        all_prompts.extend(category_prompts)
    
    print(f"Loaded {len(all_prompts)} prompts across {len(prompts_dict)} categories")
    
    # Generate comparative results
    results_df = compare_models(all_prompts, prompts_dict)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"analysis_results/bias_analysis_{timestamp}.csv", index=False)
    
    # Analyze results
    print("\nOverall Results:")
    overall_stats = results_df.groupby("model")[
        ["gender_balance", "toxicity", "diversity_score"]
    ].mean()
    print(overall_stats)
    
    # Category analysis
    print("\nCategory Analysis:")
    category_stats = results_df.groupby(["model", "prompt_category"])[
        ["gender_balance", "toxicity"]
    ].mean().reset_index()
    print(category_stats)
    
    # Generate visualizations
    generate_visualizations(results_df, timestamp)
    
    print(f"\nAnalysis complete. Results saved to analysis_results/bias_analysis_{timestamp}.csv")
    print("Visualizations saved to the analysis_results directory")