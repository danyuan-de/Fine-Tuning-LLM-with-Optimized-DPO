import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os # For handling file and directory paths
import glob # For finding files matching a specific pattern
import re

# Set plotting style
sns.set_theme(style="whitegrid")
# Attempt to set a universally available sans-serif font for broader compatibility
# Matplotlib will fall back to a default if 'Arial' is not found.
try:
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] # Added DejaVu Sans as a common fallback
    plt.rcParams['axes.unicode_minus'] = False # Resolve issue with minus sign display
except Exception as e:
    print(f"Error setting font (font might not be installed): {e}")
    print("Plot labels might not display perfectly if default font is used.")


def extract_plot_info_for_title(json_filename_base):
    """
    Extract method, file_type, lr, beta, shift, and dp from a JSON filename.
    Supports flexible presence/absence of parameters.
    """
    info = {
        "method": None,
        "file_type": None,
        "lr": None,
        "beta": None,
        "dp": None,
        "shift": None
    }

    print(f"Original filename: {json_filename_base}")

    # Remove known prefix if it exists
    cleaned_name = re.sub(r'^sum.*?Instruct_', '', json_filename_base)
    print(f"Cleaned name: {cleaned_name}")

    # Extract method
    method_match = re.search(r'^(dpop|dpo|dposhift|dpopshift)[_-]', cleaned_name, re.IGNORECASE)
    if method_match:
        info["method"] = method_match.group(1).upper()

    # Extract file_type
    file_type_match = re.search(r'_(content|structure|html|chat)[_-]', cleaned_name, re.IGNORECASE)
    if file_type_match:
        info["file_type"] = file_type_match.group(1).lower()

    # Extract learning rate
    lr_match = re.search(r'lr([0-9.e-]+)', cleaned_name, re.IGNORECASE)
    if lr_match:
        info["lr"] = lr_match.group(1)

    # Extract beta
    beta_match = re.search(r'b([0-9.]+)', cleaned_name, re.IGNORECASE)
    if beta_match:
        info["beta"] = beta_match.group(1)

    # Extract dp if exists
    dp_match = re.search(r'dp([0-9.]+)', cleaned_name, re.IGNORECASE)
    if dp_match:
        info["dp"] = dp_match.group(1)

    # Extract shift if exists
    shift_match = re.search(r'shift([0-9.]+)', cleaned_name, re.IGNORECASE)
    if shift_match:
        info["shift"] = shift_match.group(1)

    # Print debug info
    for k, v in info.items():
        print(f"{k}: {v}")

    return info


def load_data_from_json(file_path):
    """Load data from a JSON file into a Pandas DataFrame"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Assume the top level of the JSON file is a list of records
        if isinstance(data, list):
            df = pd.DataFrame(data)
        # If the top level is a dictionary and the list is under a specific key
        # elif isinstance(data, dict) and 'records' in data: # Assuming the list is under the 'records' key
        #     df = pd.DataFrame(data['records'])
        else:
            print(f"Error: Unexpected JSON structure in file {file_path}. Expected a list of records.")
            return None

        # Ensure necessary columns exist
        required_cols = [
            'ref_perplexity', 'policy_perplexity',
            'ref_chosen_perplexity', 'policy_chosen_perplexity',
            'ref_rejected_perplexity', 'policy_rejected_perplexity'
        ]
        # 'question' column is optional but good to have if it exists
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: File {file_path} is missing the following required columns: {', '.join(missing_cols)}")
            print("Attempting to continue, but some plots or statistics might fail.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not a valid JSON format")
        return None
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return None

# Modify the plotting functions to accept plot_title_prefix and an optional filename_prefix_override

def plot_perplexity_distributions(df, plot_title_prefix="", output_dir=".", filename_prefix_override=None):
    """
    Plot distributions of various perplexity metrics (box plots).
    """
    if df is None or df.empty:
        print(f"No data to plot (source: {plot_title_prefix}).") # Use plot_title_prefix for context in print
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the prefix to use for filenames
    actual_filename_prefix = filename_prefix_override if filename_prefix_override is not None else plot_title_prefix.replace(" ", "_").replace("|", "").strip("_")
    if actual_filename_prefix and not actual_filename_prefix.endswith('_'):
        actual_filename_prefix += '_'


    plot_cols_map = {
        'self_generated_ppl_comparison': (['ref_perplexity', 'policy_perplexity'], ['Reference Model PPL', 'Policy Model PPL']),
        'chosen_ppl_comparison': (['ref_chosen_perplexity', 'policy_chosen_perplexity'], ['Reference Chosen PPL', 'Policy Chosen PPL']),
        'rejected_ppl_comparison': (['ref_rejected_perplexity', 'policy_rejected_perplexity'], ['Reference Rejected PPL', 'Policy Rejected PPL'])
    }

    base_titles = {
        'self_generated_ppl_comparison': 'Reference vs Policy: Self-Generated Response PPL',
        'chosen_ppl_comparison': 'Reference vs Policy: PPL on Chosen Answers',
        'rejected_ppl_comparison': 'Reference vs Policy: PPL on Rejected Answers'
    }

    for plot_key, (cols, labels) in plot_cols_map.items():
        available_cols_for_plot = [col for col in cols if col in df.columns]
        if len(available_cols_for_plot) > 0:
            available_labels = [labels[cols.index(col)] for col in available_cols_for_plot]
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df[available_cols_for_plot])
            plt.title(f"{plot_title_prefix}{base_titles[plot_key]}") # Use the new prefix
            plt.ylabel('Perplexity')
            if available_labels:
                 plt.xticks(ticks=range(len(available_cols_for_plot)), labels=available_labels)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{actual_filename_prefix}{plot_key}.png"))
            plt.close()
        else:
            print(f"Skipping plot '{plot_title_prefix}{base_titles[plot_key]}' due to missing all required columns: {', '.join(cols)}")


def plot_chosen_vs_rejected_ppl(df, plot_title_prefix="", output_dir=".", filename_prefix_override=None):
    """
    Plot scatter plots of Chosen Answer PPL vs. Rejected Answer PPL for each model.
    """
    if df is None or df.empty:
        print(f"No data to plot (source: {plot_title_prefix}).")
        return

    os.makedirs(output_dir, exist_ok=True)
    actual_filename_prefix = filename_prefix_override if filename_prefix_override is not None else plot_title_prefix.replace(" ", "_").replace("|", "").strip("_")
    if actual_filename_prefix and not actual_filename_prefix.endswith('_'):
        actual_filename_prefix += '_'

    # Policy Model
    if 'policy_chosen_perplexity' in df.columns and 'policy_rejected_perplexity' in df.columns:
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x='policy_chosen_perplexity', y='policy_rejected_perplexity', data=df, label='Policy Model Data Points')
        df_policy_clean = df[['policy_chosen_perplexity', 'policy_rejected_perplexity']].dropna()
        if not df_policy_clean.empty:
            min_val = min(df_policy_clean['policy_chosen_perplexity'].min(), df_policy_clean['policy_rejected_perplexity'].min())
            max_val = max(df_policy_clean['policy_chosen_perplexity'].max(), df_policy_clean['policy_rejected_perplexity'].max())
            if pd.notna(min_val) and pd.notna(max_val):
                 plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='y=x (PPL Chosen = PPL Rejected)')
        plt.title(f'{plot_title_prefix}Policy Model: Chosen PPL vs. Rejected PPL')
        plt.xlabel('Perplexity on Chosen Answer')
        plt.ylabel('Perplexity on Rejected Answer')
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{actual_filename_prefix}policy_chosen_vs_rejected_ppl.png"))
        plt.close()
    else:
        print(f"Skipping Policy Model Chosen vs Rejected PPL plot due to missing 'policy_chosen_perplexity' or 'policy_rejected_perplexity' columns.")

    # Reference Model
    if 'ref_chosen_perplexity' in df.columns and 'ref_rejected_perplexity' in df.columns:
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x='ref_chosen_perplexity', y='ref_rejected_perplexity', data=df, color='orange', label='Reference Model Data Points')
        df_ref_clean = df[['ref_chosen_perplexity', 'ref_rejected_perplexity']].dropna()
        if not df_ref_clean.empty:
            min_val = min(df_ref_clean['ref_chosen_perplexity'].min(), df_ref_clean['ref_rejected_perplexity'].min())
            max_val = max(df_ref_clean['ref_chosen_perplexity'].max(), df_ref_clean['ref_rejected_perplexity'].max())
            if pd.notna(min_val) and pd.notna(max_val):
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='y=x (PPL Chosen = PPL Rejected)')
        plt.title(f'{plot_title_prefix}Reference Model: Chosen PPL vs. Rejected PPL')
        plt.xlabel('Perplexity on Chosen Answer')
        plt.ylabel('Perplexity on Rejected Answer')
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{actual_filename_prefix}ref_chosen_vs_rejected_ppl.png"))
        plt.close()
    else:
        print(f"Skipping Reference Model Chosen vs Rejected PPL plot due to missing 'ref_chosen_perplexity' or 'ref_rejected_perplexity' columns.")


def plot_ppl_differences(df, plot_title_prefix="", output_dir=".", filename_prefix_override=None):
    """
    Plot differences in PPL: Policy Model vs. Reference Model for Chosen and Rejected answers.
    Also plots Policy Model's internal difference: PPL(Rejected) - PPL(Chosen).
    """
    if df is None or df.empty:
        print(f"No data to plot (source: {plot_title_prefix}).")
        return

    os.makedirs(output_dir, exist_ok=True)
    actual_filename_prefix = filename_prefix_override if filename_prefix_override is not None else plot_title_prefix.replace(" ", "_").replace("|", "").strip("_")
    if actual_filename_prefix and not actual_filename_prefix.endswith('_'):
        actual_filename_prefix += '_'


    # Policy vs Reference: Chosen PPL difference
    if 'ref_chosen_perplexity' in df.columns and 'policy_chosen_perplexity' in df.columns:
        df['chosen_ppl_diff_policy_vs_ref'] = df['ref_chosen_perplexity'] - df['policy_chosen_perplexity']
        plt.figure(figsize=(10, 6))
        sns.histplot(df['chosen_ppl_diff_policy_vs_ref'].dropna(), kde=True)
        mean_diff_chosen = df['chosen_ppl_diff_policy_vs_ref'].mean()
        if pd.notna(mean_diff_chosen):
            plt.axvline(mean_diff_chosen, color='r', linestyle='dashed', linewidth=1, label=f'Mean Difference: {mean_diff_chosen:.2f}')
        plt.title(f'{plot_title_prefix}Chosen PPL Difference (Reference - Policy)')
        plt.xlabel('Perplexity Difference (Positive means Policy is better)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{actual_filename_prefix}chosen_ppl_difference_ref_vs_policy.png"))
        plt.close()
    else:
        print(f"Skipping Chosen PPL Difference (Ref vs Policy) plot due to missing relevant columns.")

    # Policy vs Reference: Rejected PPL difference
    if 'policy_rejected_perplexity' in df.columns and 'ref_rejected_perplexity' in df.columns:
        df['rejected_ppl_diff_policy_vs_ref'] = df['policy_rejected_perplexity'] - df['ref_rejected_perplexity']
        plt.figure(figsize=(10, 6))
        sns.histplot(df['rejected_ppl_diff_policy_vs_ref'].dropna(), kde=True)
        mean_diff_rejected = df['rejected_ppl_diff_policy_vs_ref'].mean()
        if pd.notna(mean_diff_rejected):
            plt.axvline(mean_diff_rejected, color='r', linestyle='dashed', linewidth=1, label=f'Mean Difference: {mean_diff_rejected:.2f}')
        plt.title(f'{plot_title_prefix}Rejected PPL Difference (Policy - Reference)')
        plt.xlabel('Perplexity Difference (Positive means Policy PPL for Rejected is higher)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{actual_filename_prefix}rejected_ppl_difference_policy_vs_ref.png"))
        plt.close()
    else:
        print(f"Skipping Rejected PPL Difference (Policy vs Ref) plot due to missing relevant columns.")

    # Policy Model: PPL(Rejected) - PPL(Chosen)
    if 'policy_rejected_perplexity' in df.columns and 'policy_chosen_perplexity' in df.columns:
        df['policy_rejected_minus_chosen_ppl'] = df['policy_rejected_perplexity'] - df['policy_chosen_perplexity']
        plt.figure(figsize=(10, 6))
        sns.histplot(df['policy_rejected_minus_chosen_ppl'].dropna(), kde=True)
        mean_diff_policy_internal = df['policy_rejected_minus_chosen_ppl'].mean()
        if pd.notna(mean_diff_policy_internal):
            plt.axvline(mean_diff_policy_internal, color='r', linestyle='dashed', linewidth=1, label=f'Mean Difference: {mean_diff_policy_internal:.2f}')
        plt.title(f'{plot_title_prefix}Policy Model: PPL(Rejected) - PPL(Chosen)')
        plt.xlabel('Perplexity Difference (Positive means PPL for Rejected is higher)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{actual_filename_prefix}policy_internal_ppl_difference.png"))
        plt.close()
    else:
        print(f"Skipping Policy Model internal PPL difference plot due to missing relevant columns.")

    # Reference Model: PPL(Rejected) - PPL(Chosen)
    if 'ref_rejected_perplexity' in df.columns and 'ref_chosen_perplexity' in df.columns:
        df['ref_rejected_minus_chosen_ppl'] = df['ref_rejected_perplexity'] - df['ref_chosen_perplexity']
        plt.figure(figsize=(10, 6))
        sns.histplot(df['ref_rejected_minus_chosen_ppl'].dropna(), kde=True)
        mean_diff_ref_internal = df['ref_rejected_minus_chosen_ppl'].mean()
        if pd.notna(mean_diff_ref_internal):
            plt.axvline(mean_diff_ref_internal, color='g', linestyle='dashed', linewidth=1, label=f'Mean Difference: {mean_diff_ref_internal:.2f}') # Changed color for distinction
        plt.title(f'{plot_title_prefix}Reference Model: PPL(Rejected) - PPL(Chosen)')
        plt.xlabel('Perplexity Difference (Positive means PPL for Rejected is higher)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        # Ensure filename_prefix_override is handled correctly or generate a suitable one
        actual_filename_prefix_ref = filename_prefix_override if filename_prefix_override is not None else plot_title_prefix.replace(" ", "_").replace("|", "").strip("_")
        if actual_filename_prefix_ref and not actual_filename_prefix_ref.endswith('_'):
            actual_filename_prefix_ref += '_'
        plt.savefig(os.path.join(output_dir, f"{actual_filename_prefix_ref}ref_internal_ppl_difference.png"))
        plt.close()
    else:
        print(f"Skipping Reference Model internal PPL difference plot due to missing relevant columns.")
    """
    Plot differences in PPL: Policy Model vs. Reference Model for Chosen and Rejected answers.
    Also plots Policy Model's internal difference: PPL(Rejected) - PPL(Chosen).
    """
    if df is None or df.empty:
        print(f"No data to plot (source: {actual_filename_prefix}).")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Policy vs Reference: Chosen PPL difference
    if 'ref_chosen_perplexity' in df.columns and 'policy_chosen_perplexity' in df.columns:
        df['chosen_ppl_diff_policy_vs_ref'] = df['ref_chosen_perplexity'] - df['policy_chosen_perplexity']
        plt.figure(figsize=(10, 6))
        sns.histplot(df['chosen_ppl_diff_policy_vs_ref'].dropna(), kde=True) # dropna() to handle missing values
        mean_diff_chosen = df['chosen_ppl_diff_policy_vs_ref'].mean()
        if pd.notna(mean_diff_chosen):
            plt.axvline(mean_diff_chosen, color='r', linestyle='dashed', linewidth=1, label=f'Mean Difference: {mean_diff_chosen:.2f}')
        plt.title(f'{actual_filename_prefix}Chosen PPL Difference (Reference - Policy)')
        plt.xlabel('Perplexity Difference (Positive means Policy is better)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{actual_filename_prefix}chosen_ppl_difference_ref_vs_policy.png"))
        plt.close()
    else:
        print(f"Skipping Chosen PPL Difference (Ref vs Policy) plot due to missing relevant columns.")

    # Policy vs Reference: Rejected PPL difference
    if 'policy_rejected_perplexity' in df.columns and 'ref_rejected_perplexity' in df.columns:
        df['rejected_ppl_diff_policy_vs_ref'] = df['policy_rejected_perplexity'] - df['ref_rejected_perplexity']
        plt.figure(figsize=(10, 6))
        sns.histplot(df['rejected_ppl_diff_policy_vs_ref'].dropna(), kde=True)
        mean_diff_rejected = df['rejected_ppl_diff_policy_vs_ref'].mean()
        if pd.notna(mean_diff_rejected):
            plt.axvline(mean_diff_rejected, color='r', linestyle='dashed', linewidth=1, label=f'Mean Difference: {mean_diff_rejected:.2f}')
        plt.title(f'{actual_filename_prefix}Rejected PPL Difference (Policy - Reference)')
        plt.xlabel('Perplexity Difference (Positive means Policy PPL for Rejected is higher)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{actual_filename_prefix}rejected_ppl_difference_policy_vs_ref.png"))
        plt.close()
    else:
        print(f"Skipping Rejected PPL Difference (Policy vs Ref) plot due to missing relevant columns.")


    # Policy Model: PPL(Rejected) - PPL(Chosen)
    if 'policy_rejected_perplexity' in df.columns and 'policy_chosen_perplexity' in df.columns:
        df['policy_rejected_minus_chosen_ppl'] = df['policy_rejected_perplexity'] - df['policy_chosen_perplexity']
        plt.figure(figsize=(10, 6))
        sns.histplot(df['policy_rejected_minus_chosen_ppl'].dropna(), kde=True)
        mean_diff_policy_internal = df['policy_rejected_minus_chosen_ppl'].mean()
        if pd.notna(mean_diff_policy_internal):
            plt.axvline(mean_diff_policy_internal, color='r', linestyle='dashed', linewidth=1, label=f'Mean Difference: {mean_diff_policy_internal:.2f}')
        plt.title(f'{actual_filename_prefix}Policy Model: PPL(Rejected) - PPL(Chosen)')
        plt.xlabel('Perplexity Difference (Positive means PPL for Rejected is higher)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{actual_filename_prefix}policy_internal_ppl_difference.png"))
        plt.close()
    else:
        print(f"Skipping Policy Model internal PPL difference plot due to missing relevant columns.")

# --- Main Program ---
# --- Main Program ---
if __name__ == "__main__":
    # === Configure input and output directories ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results')
    output_charts_folder = os.path.join(script_dir, '..', 'charts_output')
    
    if not os.path.exists(output_charts_folder):
        os.makedirs(output_charts_folder)
        print(f"Created output directory: {output_charts_folder}")

    if not os.path.isdir(results_dir):
        print(f"Error: Input directory '{results_dir}' does not exist. Please create it and place JSON files inside, or modify the path.")
        exit()

    json_files = glob.glob(os.path.join(results_dir, '*generated*.json'))

    if not json_files:
        print(f"No JSON files containing 'generated' found in directory '{results_dir}'.")
        exit()

    print(f"Found {len(json_files)} JSON files. Processing and generating charts for each...")

    perplexity_cols_for_stats = [
        'ref_perplexity', 'policy_perplexity',
        'ref_chosen_perplexity', 'policy_chosen_perplexity',
        'ref_rejected_perplexity', 'policy_rejected_perplexity'
    ]

    for json_file_path in json_files:
        print(f"\n--- Processing file: {json_file_path} ---")
        df_current_file = load_data_from_json(json_file_path)

        if df_current_file is not None and not df_current_file.empty:
            base_name = os.path.basename(json_file_path)
            extracted_info_dict = extract_plot_info_for_title(os.path.splitext(base_name)[0]) # Pass filename without extension

            # --- MODIFICATION START ---
            # Create a more readable string from the extracted info for file prefixes and titles
            # This will filter out None values and join the existing ones.
            info_parts = [f"{k}_{v}" for k, v in extracted_info_dict.items() if v is not None]
            # If you want a specific order or format, adjust this list comprehension and join
            # Example: prioritize method, lr, beta
            ordered_info_parts = []
            if extracted_info_dict.get("method"):
                ordered_info_parts.append(extracted_info_dict["method"])
            if extracted_info_dict.get("file_type"):
                ordered_info_parts.append(extracted_info_dict["file_type"])
            if extracted_info_dict.get("lr"):
                ordered_info_parts.append(f"lr{extracted_info_dict['lr']}")
            if extracted_info_dict.get("beta"):
                ordered_info_parts.append(f"b{extracted_info_dict['beta']}")
            if extracted_info_dict.get("dp"):
                ordered_info_parts.append(f"dp{extracted_info_dict['dp']}")
            if extracted_info_dict.get("shift"):
                 ordered_info_parts.append(f"shift{extracted_info_dict['shift']}")
            
            readable_file_prefix = "_".join(ordered_info_parts) + "_" if ordered_info_parts else os.path.splitext(base_name)[0] + '_'
            title_suffix = " | " + " ".join(ordered_info_parts) if ordered_info_parts else ""
            # --- MODIFICATION END ---

            file_specific_output_dir = os.path.join(output_charts_folder, os.path.splitext(base_name)[0])
            os.makedirs(file_specific_output_dir, exist_ok=True)

            print(f"\nDescriptive Statistics:")
            existing_ppl_cols = [col for col in perplexity_cols_for_stats if col in df_current_file.columns]
            if existing_ppl_cols:
                print(df_current_file[existing_ppl_cols].describe())
            else:
                print("No specified perplexity-related columns found in this file for descriptive statistics.")
            
            print(f"\nPlotting charts for the file, saving to '{file_specific_output_dir}'...")
            # Pass the readable_file_prefix or title_suffix to the plotting functions
            # For plot titles, it's better to pass the title_suffix and append it.
            # For filenames, use readable_file_prefix.

            # Modify how titles are generated in plotting functions or pass the suffix
            # Let's modify the plotting functions to accept a title_suffix
            
            # We need to pass `readable_file_prefix` for filenames
            # and `title_suffix` to be appended to the original titles in plotting functions.
            # For simplicity, I'll slightly modify how titles are constructed in the main loop here,
            # and pass the full desired title prefix to the plotting functions.
            
            plot_title_prefix = " ".join(ordered_info_parts) + " " if ordered_info_parts else ""


            plot_perplexity_distributions(df_current_file, plot_title_prefix, file_specific_output_dir, readable_file_prefix)
            plot_chosen_vs_rejected_ppl(df_current_file, plot_title_prefix, file_specific_output_dir, readable_file_prefix)
            plot_ppl_differences(df_current_file, plot_title_prefix, file_specific_output_dir, readable_file_prefix)
            print(f"Charts saved.")

            df_current_file['source_file'] = base_name
            # all_dataframes.append(df_current_file) # Uncomment to collect data for aggregate analysis
        else:
            print(f"Could not process file {json_file_path} or file is empty.")
    # ... (rest of your main program, including aggregate analysis if uncommented) ...
    print("\n\nProcessing complete!")