import os
import sys
import glob
import json
import matplotlib.pyplot as plt

def process_file(json_path: str, plot_dir: str):
    """
    Read a benchmark JSON file, plot Original vs. Fine-tuned model accuracies,
    and save the figure as a PNG in the given directory.
    """
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Expecting structure: { "subject1": {"ori_acc": x, "ft_acc": y}, ... }
    subjects = list(data.keys())
    ori_acc = [data[subj]['ori_acc'] for subj in subjects]
    ft_acc  = [data[subj]['ft_acc']  for subj in subjects]

    # Set up bar chart
    x_coords = range(len(subjects)) # Renamed x to x_coords for clarity
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7)) # Increased figure size for better readability
    rects1 = ax.bar([i - width/2 for i in x_coords], ori_acc, width, label='Original')
    rects2 = ax.bar([i + width/2 for i in x_coords], ft_acc,  width, label='Fine-tuned')

    # Configure axes and labels
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Benchmark: {os.path.basename(json_path)}', fontsize=14)
    ax.set_xticks(x_coords)
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(ori_acc, default=0), max(ft_acc, default=0)) * 1.15) # Adjust y-limit for text space

    # Function to add labels on top of bars
    def autolabel(rects, acc_values):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            # Format to 2 decimal places if it's a float, otherwise keep as is
            label_text = f'{acc_values[i]:.3f}' if isinstance(acc_values[i], float) else str(acc_values[i])
            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Add labels to bars
    autolabel(rects1, ori_acc)
    autolabel(rects2, ft_acc)

    fig.tight_layout() # Call tight_layout after adding annotations

    # Save plot as PNG with same base name as JSON
    base_name = os.path.basename(json_path)
    png_name  = os.path.splitext(base_name)[0] + '.png'
    output_path = os.path.join(plot_dir, png_name)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    """
    Search for all JSON files under '../results' whose filenames contain 'benchmark',
    process each one, and save plots under '../resultsChart/benchmarkPlots'.
    """
    results_dir = '../results'

    # 1. Recursively find all JSON files that include 'benchmark' in their name
    pattern = os.path.join(results_dir, '**', '*benchmark*.json')
    file_list = glob.glob(pattern, recursive=True)
    if not file_list:
        print("No JSON files found containing 'benchmark' in their names.")
        sys.exit(1)
    print(f"Found {len(file_list)} JSON files to process.")

    # 2. Ensure the output directory exists
    plot_dir = '../resultsChart/benchmarkPlots'
    os.makedirs(plot_dir, exist_ok=True)

    # 3. Process each file
    for json_path in file_list:
        print(f"Processing {json_path} ...", end=' ')
        try:
            process_file(json_path, plot_dir)
            print("Done.")
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == '__main__':
    main()
