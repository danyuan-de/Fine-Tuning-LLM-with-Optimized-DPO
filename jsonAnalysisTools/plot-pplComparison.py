import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import re

# --- Configuration ---
BASE_RESULTS_PATH = "../results/"
output_charts_folder_base = os.path.join("../resultsChart/ppl-chartsDiagAnnot") # New folder
os.makedirs(output_charts_folder_base, exist_ok=True)

# --- Helper Functions --- (Assume get_lr_from_filename_str, extract_plot_info_from_filename, load_data_from_json are unchanged)
def get_lr_from_filename_str(filename_str):
    match = re.search(r'lr([0-9.e-]+)', filename_str)
    return match.group(1) if match else "N/A"

def extract_plot_info_from_filename(json_filename_base, current_dataset_type_for_fallback):
    info = { "method": None, "file_type": None, "lr": None, "beta": None, "dp": None, "shift": None }
    cleaned_name = re.sub(r'^sum_?[A-Za-z0-9.-]+?Instruct_', '', json_filename_base, flags=re.IGNORECASE)
    if cleaned_name == json_filename_base:
        cleaned_name = re.sub(r'^sum_?[A-Za-z0-9.-]+?_', '', json_filename_base, flags=re.IGNORECASE)
    file_type_match = re.search(r'_(content|structure|html|chat)_', cleaned_name, re.IGNORECASE)
    if file_type_match:
        info["file_type"] = file_type_match.group(1).lower()
        cleaned_name = cleaned_name.replace(f"_{info['file_type']}_", "_", 1)
    else:
        info["file_type"] = current_dataset_type_for_fallback
        if current_dataset_type_for_fallback and f"_{current_dataset_type_for_fallback}_" in cleaned_name:
            cleaned_name = cleaned_name.replace(f"_{current_dataset_type_for_fallback}_", "_", 1)
    info["lr"] = get_lr_from_filename_str(json_filename_base)
    parts = cleaned_name.split('_')
    if parts:
        method_match = re.search(r'^(DPO|DPOP|DPOSHIFT)', parts[0], re.IGNORECASE)
        if method_match: info["method"] = method_match.group(1).upper()
    beta_match = re.search(r'b([0-9.]+)', cleaned_name, re.IGNORECASE)
    if beta_match: info["beta"] = beta_match.group(1)
    return info

def load_data_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        if isinstance(data, list) and len(data) > 0: df = pd.DataFrame(data)
        elif isinstance(data, dict):
            list_key = next((key for key, value in data.items() if isinstance(value, list)), None)
            if list_key: df = pd.DataFrame(data[list_key])
            else: return None
        else: return None
        return df
    except FileNotFoundError: return None
    except json.JSONDecodeError: return None
    except Exception: return None


# --- Plotting Function for Combined PPL Distributions ---
def plot_combined_ppl_distributions(config_data_map_with_lrs, dataset_type, output_dir_base, 
                                    dynamic_plot_order, dynamic_plot_colors):
    if not config_data_map_with_lrs:
        return

    first_policy_label = next(iter(config_data_map_with_lrs.keys()))
    ref_df_source = config_data_map_with_lrs[first_policy_label]

    ppl_categories = {
        "Chosen": ("ref_chosen_perplexity", "policy_chosen_perplexity", "PPL on Chosen Responses", "Perplexity (Lower is Better)"),
        "Rejected": ("ref_rejected_perplexity", "policy_rejected_perplexity", "PPL on Rejected Responses", "Perplexity (Higher is Better)"),
        "Self-Generated": ("ref_perplexity", "policy_perplexity", "Self-Generated Response PPL", "Perplexity (Lower is Better, closer to Ref)")
    }
    all_mean_ppl_values = {} 

    for key_name, (ref_col, policy_col, title_suffix, y_label) in ppl_categories.items():
        all_plot_data_dfs = [] 
        if ref_col in ref_df_source.columns:
            ref_data = ref_df_source[ref_col].dropna()
            if not ref_data.empty:
                all_plot_data_dfs.append(pd.DataFrame({'Perplexity': ref_data, 'Method': 'Reference'}))
                if "Reference" not in all_mean_ppl_values: all_mean_ppl_values["Reference"] = {}
                all_mean_ppl_values["Reference"][f"Ref_{key_name}_PPL"] = round(ref_data.mean(), 3)
        for descriptive_label, df_policy in config_data_map_with_lrs.items():
            method_label_for_plot = f"{descriptive_label} Policy"
            if policy_col in df_policy.columns:
                policy_data = df_policy[policy_col].dropna()
                if not policy_data.empty:
                    all_plot_data_dfs.append(pd.DataFrame({'Perplexity': policy_data, 'Method': method_label_for_plot}))
                    if descriptive_label not in all_mean_ppl_values: all_mean_ppl_values[descriptive_label] = {}
                    all_mean_ppl_values[descriptive_label][f"Policy_{key_name}_PPL"] = round(policy_data.mean(), 3)
        
        if not all_plot_data_dfs: plt.close(); continue
        long_form_df = pd.concat(all_plot_data_dfs, ignore_index=True)
        long_form_df['Method'] = pd.Categorical(long_form_df['Method'], categories=dynamic_plot_order, ordered=True)
        if long_form_df.empty or long_form_df['Perplexity'].isnull().all(): plt.close(); continue
        
        plot_order_with_data = [m for m in dynamic_plot_order if m in long_form_df['Method'].unique() and not long_form_df[long_form_df['Method'] == m]['Perplexity'].isnull().all()]
        if not plot_order_with_data: plt.close(); continue

        num_total_boxes = len(plot_order_with_data)
        # Adjusted width calculation slightly if needed for side annotations
        fig_width = max(8, 2 + num_total_boxes * 2.8) # Increased multiplier for width
        plt.figure(figsize=(fig_width, 8)) # Height can be less aggressive now

        try:
            ax = sns.boxplot(x='Method', y='Perplexity', data=long_form_df, 
                             palette=dynamic_plot_colors, order=plot_order_with_data, width=0.7) 
            
            # print(f"[{dataset_type}] DEBUG PLOT: After sns.boxplot for '{key_name}': len(ax.artists)={len(ax.artists)}, len(ax.patches)={len(ax.patches)}")

            plt.xticks(rotation=15, ha="right", fontsize=9) # Rotation can be less aggressive too
            base_dt_for_title = dataset_type.split('_lr')[0].capitalize()
            title_str = f"Combined {title_suffix} ({base_dt_for_title} Dataset)"
            plt.title(title_str, fontsize=14, pad=15) 
            plt.ylabel(y_label, fontsize=12)
            plt.xlabel("") 
            plt.tick_params(axis='y', labelsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plotted_tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
            plot_ymin, plot_ymax = ax.get_ylim()
            y_range = plot_ymax - plot_ymin
            if y_range == 0: y_range = 1 

            for i, method_on_plot_axis in enumerate(plotted_tick_labels):
                current_box_data = long_form_df[long_form_df['Method'] == method_on_plot_axis]['Perplexity'].dropna()
                
                if not current_box_data.empty:
                    n_obs = len(current_box_data)
                    mean_val = current_box_data.mean()
                    median_val = current_box_data.median()
                    q1_val = current_box_data.quantile(0.25)
                    q3_val = current_box_data.quantile(0.75)

                    # --- New Text Placement Logic ---
                    # X-coordinate: to the right of the box
                    # Box center is at 'i', box width is 0.7, so right edge is at i + 0.35
                    # Place text starting slightly to the right of the box's right edge.
                    text_x_pos = i + (0.7 / 2) + 0.05 

                    # Y-coordinate: bottom of text slightly above Q3
                    if pd.notna(q3_val):
                        text_anchor_y = q3_val
                    elif pd.notna(median_val): # Fallback if q3 is NaN
                        text_anchor_y = median_val
                    else: # Further fallback
                        text_anchor_y = plot_ymin
                    
                    text_y_pos = text_anchor_y + (y_range * 0.015) # Small nudge above Q3/anchor

                    # Basic y-axis bounds check for text_y_pos
                    text_y_pos = max(text_y_pos, plot_ymin + (y_range * 0.02))
                    # Ensure bottom of text block does not start too high on the axis
                    # (e.g., not in the top 5% of the axis range to leave room for text height)
                    text_y_pos = min(text_y_pos, plot_ymax - (y_range * 0.05))


                    annotation_text = (
                        f"N: {n_obs}\nMean: {mean_val:.2f}\nMed: {median_val:.2f}\n"
                        f"Q3: {q3_val:.2f}\nQ1: {q1_val:.2f}"
                    )
                    
                    if not (pd.isna(text_x_pos) or pd.isna(text_y_pos)): 
                        ax.text(text_x_pos, text_y_pos, annotation_text, 
                                horizontalalignment='left', # Align left edge of text at text_x_pos
                                verticalalignment='bottom', # Align bottom edge of text at text_y_pos
                                fontsize=6, color='black', 
                                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.80, ec='darkgrey')) # Slightly more opaque bbox
            
            # Adjust layout: give reasonable margins. Top margin can be less aggressive now.
            # rect: [left, bottom, right, top]
            plt.tight_layout(rect=[0.05, 0.08, 0.95, 0.92]) 

            filename_parts = ["combined", dataset_type, f"{key_name.lower().replace('-', '_').replace(' ', '_')}_ppl_diag_annot.png"]
            filename = "_".join(filename_parts)
            filename = re.sub(r'[^\w\.\-_]', '_', filename) 
            save_path = os.path.join(output_dir_base, filename)
            plt.savefig(save_path, dpi=300) 
            print(f"[{dataset_type}] Saved diagonal annotation plot: {save_path}")
        except Exception as e:
            print(f"[{dataset_type}] CRITICAL ERROR during plotting diagonal annotation for {key_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close() 
    
    if all_mean_ppl_values: # JSON saving part remains unchanged
        output_json_filename = f"all_mean_ppl_values_{dataset_type}.json" 
        output_json_path = os.path.join(output_dir_base, output_json_filename)
        try:
            with open(output_json_path, "w") as f: json.dump(all_mean_ppl_values, f, indent=4)
        except Exception as e: print(f"[{dataset_type}] Error saving mean PPL values to JSON: {e}")

# --- Main Program (ensure this is the same as your last working version) ---
if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Verdana'] 
        plt.rcParams['axes.unicode_minus'] = False 
    except Exception as e:
        print(f"Error setting Matplotlib font: {e}. Matplotlib might use a default font.")

    all_json_files_in_path = glob.glob(os.path.join(BASE_RESULTS_PATH, "*.json"))
    dataset_type_pattern = re.compile(
        r'sum_Llama-3\.1-8B-Instruct_(?:DPO|DPOP|DPOSHIFT|DPOPSHIFT)_([a-zA-Z]+)_generated_output_.*\.json'
    )
    found_base_dataset_types = set() 
    for f_path in all_json_files_in_path:
        f_name = os.path.basename(f_path)
        match = dataset_type_pattern.search(f_name)
        if match:
            dt = match.group(1).lower()
            if dt in ["content", "html", "structure", "chat"]: 
                 found_base_dataset_types.add(dt)

    if not found_base_dataset_types:
        print(f"No specified base dataset types found in {BASE_RESULTS_PATH}")
        exit()
    print(f"Found base dataset types to process: {sorted(list(found_base_dataset_types))}")

    lrs_to_process = ["3.0e-06", "5.0e-07"] 
    for current_base_dataset_type in sorted(list(found_base_dataset_types)):
        for current_lr_str_value in lrs_to_process:
            dataset_type_with_lr = f"{current_base_dataset_type}_lr{current_lr_str_value}" 
            print(f"\n=== Processing Dataset Type: {current_base_dataset_type.upper()} with LR: {current_lr_str_value} ===")
            
            dpo_base_filename = f"sum_Llama-3.1-8B-Instruct_DPO_{current_base_dataset_type}_generated_output_lr{current_lr_str_value}_b0.30.json"
            dpop_base_filename = f"sum_Llama-3.1-8B-Instruct_DPOP_{current_base_dataset_type}_generated_output_lr{current_lr_str_value}_b0.30_dp50.0.json"
            dposhift_base_filename = f"sum_Llama-3.1-8B-Instruct_DPOSHIFT_{current_base_dataset_type}_generated_output_lr{current_lr_str_value}_b0.30_shift0.75.json"

            lr_dpo = get_lr_from_filename_str(dpo_base_filename) 
            lr_dpop = get_lr_from_filename_str(dpop_base_filename) 
            lr_dposhift = get_lr_from_filename_str(dposhift_base_filename) 
            
            if not (lr_dpo == current_lr_str_value and lr_dpop == current_lr_str_value and lr_dposhift == current_lr_str_value):
                print(f"[{dataset_type_with_lr}] Mismatch in expected LR ({current_lr_str_value}). Skipping.")
                continue

            label_dpo_dynamic = f"DPO (β=0.30, LR={lr_dpo})"
            label_dpop_dynamic = f"DPOP (β=0.30, dp=50, LR={lr_dpop})"
            label_dposhift_dynamic = f"DPOSHIFT (β=0.30, shift=0.75, LR={lr_dposhift})"
            
            target_configs_with_filenames = {
                label_dpo_dynamic: dpo_base_filename,
                label_dpop_dynamic: dpop_base_filename,
                label_dposhift_dynamic: dposhift_base_filename,
            }
            dynamic_plot_order_for_func = [
                "Reference",
                f"{label_dpo_dynamic} Policy",
                f"{label_dpop_dynamic} Policy",
                f"{label_dposhift_dynamic} Policy"
            ]
            dynamic_plot_colors_for_func = {
                "Reference": "grey",
                f"{label_dpo_dynamic} Policy": "dodgerblue",
                f"{label_dpop_dynamic} Policy": "orangered",
                f"{label_dposhift_dynamic} Policy": "forestgreen",
            }
            loaded_data_map_for_plotting = {}
            expected_configs_to_load = len(target_configs_with_filenames)
            successfully_loaded_configs = 0
            for descriptive_label, filename_to_load in target_configs_with_filenames.items():
                full_file_path = os.path.join(BASE_RESULTS_PATH, filename_to_load)
                df_loaded = load_data_from_json(full_file_path)
                if df_loaded is not None and not df_loaded.empty:
                    loaded_data_map_for_plotting[descriptive_label] = df_loaded
                    successfully_loaded_configs += 1
                else:
                    print(f"[{dataset_type_with_lr}] Failed to load or data is empty for: {descriptive_label} from {full_file_path}")
            
            if loaded_data_map_for_plotting and successfully_loaded_configs == expected_configs_to_load:
                plot_combined_ppl_distributions(
                    config_data_map_with_lrs=loaded_data_map_for_plotting,
                    dataset_type=dataset_type_with_lr, 
                    output_dir_base=output_charts_folder_base, 
                    dynamic_plot_order=dynamic_plot_order_for_func,
                    dynamic_plot_colors=dynamic_plot_colors_for_func
                )
            else:
                print(f"[{dataset_type_with_lr}] Could not load data for ALL {expected_configs_to_load} configurations. Skipping PPL plots.")
    print("\nProcessing complete for all detected dataset types and LRs!")