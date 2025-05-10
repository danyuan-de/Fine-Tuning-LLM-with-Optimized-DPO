import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# CHOOSE THE DATASET TYPE YOU WANT TO PLOT: "content", "structure", or "html"
DATASET_TYPE_TO_PLOT = "structure" # Options: "content", "structure", "html"
# file_prefix = "sum_Llama-3.1-8B-Instruct_"
# Define the base path to your CSV files directory
# Modify this to your local directory structure if needed
BASE_PATH = "../results/"  # Example: 

# Define the primary configurations and their corresponding file name patterns
# The {dataset_type} placeholder will be replaced by DATASET_TYPE_TO_PLOT
primary_configs_patterns = {
    "DPO (β=0.30, lr=3e-6)": "sum_Llama-3.1-8B-Instruct_DPO_{dataset_type}_lr3.0e-06_b0.30.csv",
    "DPOP (β=0.30, dp=50, lr=3e-6)": "sum_Llama-3.1-8B-Instruct_DPOP_{dataset_type}_lr3.0e-06_b0.30_dp50.0.csv",
    "DPOSHIFT (β=0.30, shift=0.75, lr=3e-6)": "sum_Llama-3.1-8B-Instruct_DPOSHIFT_{dataset_type}_lr3.0e-06_b0.30_shift0.75.csv",
    # Add other configurations here if needed
    "DPO (β=0.30, lr=5e-7)": "sum_Llama-3.1-8B-Instruct_DPO_{dataset_type}_lr5.0e-07_b0.30.csv",
    "DPOP (β=0.30, dp=50, lr=5e-7)": "sum_Llama-3.1-8B-Instruct_DPOP_{dataset_type}_lr5.0e-07_b0.30_dp50.0.csv",
    "DPOSHIFT (β=0.30, shift=0.75, lr=5e-7)": "sum_Llama-3.1-8B-Instruct_DPOSHIFT_{dataset_type}_lr5.0e-07_b0.30_shift0.75.csv",
}

# Define colors, linestyles, and markers for consistent plotting
config_styles = {
    "DPO (β=0.30, lr=3e-6)": {"color": "dodgerblue", "linestyle": "-", "marker": "o"},
    "DPOP (β=0.30, dp=50, lr=3e-6)": {"color": "orangered", "linestyle": "-", "marker": "s"},
    "DPOSHIFT (β=0.30, shift=0.75, lr=3e-6)": {"color": "forestgreen", "linestyle": "-", "marker": "^"},
    # Add styles for other configs if you decide to plot them
    "DPO (β=0.30, lr=5e-7)": {"color": "lightskyblue", "linestyle": "--", "marker": "x"},
    "DPOP (β=0.30, dp=50, lr=5e-7)": {"color": "tomato", "linestyle": "--", "marker": "p"},
    "DPOSHIFT (β=0.30, shift=0.75, lr=5e-7)": {"color": "limegreen", "linestyle": "--", "marker": "d"},
}

# --- Load Data ---
all_dataframes = {}
print(f"Attempting to load data for dataset type: '{DATASET_TYPE_TO_PLOT}'...")

for config_label, file_pattern in primary_configs_patterns.items():
    file_name = file_pattern.format(dataset_type=DATASET_TYPE_TO_PLOT)
    file_path = os.path.join(BASE_PATH, file_name)
    try:
        df = pd.read_csv(file_path)
        all_dataframes[config_label] = df
        print(f"Successfully loaded: {file_path} as {config_label}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Skipping {config_label}.")
        all_dataframes[config_label] = pd.DataFrame() # Add empty DataFrame to avoid key errors
    except Exception as e:
        print(f"Error processing file {file_path} for {config_label}: {e}")
        all_dataframes[config_label] = pd.DataFrame()

# --- Plotting ---
if not any(not df.empty for df in all_dataframes.values()):
    print(f"No data loaded for dataset type '{DATASET_TYPE_TO_PLOT}'. Skipping plot generation.")
else:
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False) # Reduced width slightly as legend is removed

    # Plot val_reward_margin
    ax1 = axes[0]
    for config_label, df in all_dataframes.items():
        if 'step' in df.columns and 'val_reward_margin' in df.columns and not df.empty:
            style = config_styles.get(config_label, {})
            line, = ax1.plot(df['step'], df['val_reward_margin'],
                             # No label here for individual lines if adding text annotations
                             color=style.get('color'),
                             linestyle=style.get('linestyle'),
                             marker=style.get('marker'),
                             markersize=4, # Slightly smaller markers
                             linewidth=2)
            # Add text annotation at the end of the line
            if not df.empty:
                last_step = df['step'].iloc[-1]
                last_value = df['val_reward_margin'].iloc[-1]
                ax1.text(last_step + 0.5, last_value, # Add slight offset to x for clarity
                         f" {config_label}", # Add a space for padding
                         color=style.get('color', 'black'),
                         verticalalignment='center', # Adjust alignment as needed
                         fontsize=9)


    ax1.set_ylabel('Validation Reward Margin', fontsize=12)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_title(f'Validation Reward Margin vs. Training Steps ({DATASET_TYPE_TO_PLOT.capitalize()} Dataset)', fontsize=14)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    # ax1.legend(loc='best', fontsize='small') # Legend removed in favor of direct labels

    # Plot val_loss
    ax2 = axes[1]
    for config_label, df in all_dataframes.items():
        if 'step' in df.columns and 'val_loss' in df.columns and not df.empty:
            style = config_styles.get(config_label, {})
            line, = ax2.plot(df['step'], df['val_loss'],
                             # No label here
                             color=style.get('color'),
                             linestyle=style.get('linestyle'),
                             marker=style.get('marker'),
                             markersize=4,
                             linewidth=2)
            # Add text annotation at the end of the line
            if not df.empty:
                last_step = df['step'].iloc[-1]
                last_value = df['val_loss'].iloc[-1]
                ax2.text(last_step + 0.5, last_value,
                         f" {config_label}",
                         color=style.get('color', 'black'),
                         verticalalignment='center', # 'bottom' or 'top' might work better for val_loss
                         fontsize=9)

    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title(f'Validation Loss vs. Training Steps ({DATASET_TYPE_TO_PLOT.capitalize()} Dataset)', fontsize=14)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    # ax2.legend(loc='best', fontsize='small') # Legend removed

    ax3 = axes[2]
    for config_label, df in all_dataframes.items():
        if 'step' in df.columns and 'train_loss' in df.columns and not df.empty:
            style = config_styles.get(config_label, {})
            line, = ax3.plot(df['step'], df['train_loss'],
                             # No label here
                             color=style.get('color'),
                             linestyle=style.get('linestyle'),
                             marker=style.get('marker'),
                             markersize=4,
                             linewidth=2)
            # Add text annotation at the end of the line
            if not df.empty:
                last_step = df['step'].iloc[-1]
                last_value = df['train_loss'].iloc[-1]
                ax3.text(last_step + 0.5, last_value,
                         f" {config_label}",
                         color=style.get('color', 'black'),
                         verticalalignment='center', # 'bottom' or 'top' might work better for val_loss
                         fontsize=9)

    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.set_ylabel('Training Loss', fontsize=12)
    ax3.set_title(f'Training Loss vs. Training Steps ({DATASET_TYPE_TO_PLOT.capitalize()} Dataset)', fontsize=14)
    ax3.tick_params(axis='both', labelsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    # --- Save Figure ---
    output_folder = os.path.join("../loss_margin")
    os.makedirs(output_folder, exist_ok=True)
    if len(primary_configs_patterns) == 3:
        output_plot_filename = f"{output_folder}/{DATASET_TYPE_TO_PLOT}_reward_loss_comparison.png"
    else:
        output_plot_filename = f"{output_folder}/all_{DATASET_TYPE_TO_PLOT}_reward_loss_comparison.png"
    plt.savefig(output_plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_plot_filename}")
    plt.show()