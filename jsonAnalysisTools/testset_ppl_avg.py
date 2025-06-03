import json
import glob # To find files matching a pattern
import os   # To manipulate filenames (e.g., get base name, change extension)
import csv  # To write data in CSV format

# Step 1: Find relevant JSON files
# Use glob to find all files in the current directory that contain 'generated' in their name and end with .json
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, '..', 'results')
json_files = glob.glob(os.path.join(results_dir, '*generated*.json'))

if not json_files:
    print("No JSON files with 'generated' in their name were found in the current directory.")
else:
    print(f"Found {len(json_files)} JSON file(s) to process: {json_files}")

# Step 2: Loop through each found JSON file
for json_file_path in json_files:
    print(f"\nProcessing file: {json_file_path}...")

    try:
        # Step 2a: Read and parse the JSON data from the current file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        # This case should ideally not be reached if glob is used correctly, but good for robustness
        print(f"  Error: The file '{json_file_path}' was not found during processing.")
        continue # Skip to the next file if this one can't be found
    except json.JSONDecodeError:
        print(f"  Error: The file '{json_file_path}' is not a valid JSON file. It might be corrupted.")
        continue # Skip to the next file if JSON is malformed
    except Exception as e:
        print(f"  An unexpected error occurred while reading {json_file_path}: {e}")
        continue # Skip to the next file on other unexpected errors

    # Ensure the loaded data is a list (as expected from the structure) and not empty
    if not isinstance(data, list) or not data:
        print(f"  Warning: The file '{json_file_path}' does not contain a list of entries or is empty. Skipping.")
        continue

    # Step 2b: Calculate averages for the data in the current JSON file
    # Initialize dictionaries to store sums of perplexity scores
    sums = {
        "ref_perplexity": 0,
        "policy_perplexity": 0,
        "ref_chosen_perplexity": 0,
        "policy_chosen_perplexity": 0,
        "ref_rejected_perplexity": 0,
        "policy_rejected_perplexity": 0
    }
    valid_entries_count = 0 # Counter for entries that are valid for averaging

    # Iterate through each entry (dictionary) in the JSON data list
    for entry in data:
        if not isinstance(entry, dict):
            # print(f"  Warning: Skipping a non-dictionary item in {json_file_path}")
            continue # Skip if an item in the list is not a dictionary

        # Check if all required keys are present and their values are numeric for this entry
        is_entry_fully_valid = True
        temp_values_to_add = {} # Temporarily store values for the current entry

        for key in sums.keys():
            if key not in entry or not isinstance(entry[key], (int, float)):
                is_entry_fully_valid = False
                # Optional: print a warning for more detailed debugging if an entry is skipped
                # print(f"  Warning: Entry in {json_file_path} is missing key '{key}' or its value is not numeric. Skipping this entry for averaging.")
                break # Stop checking other keys for this entry; it's invalid

            temp_values_to_add[key] = entry[key] # Store valid value

        if is_entry_fully_valid:
            # If the entry is fully valid, add its values to the sums
            for key in sums.keys():
                sums[key] += temp_values_to_add[key]
            valid_entries_count += 1 # Increment the count of valid entries

    # Calculate averages if there were any valid entries
    if valid_entries_count > 0:
        averages = {}
        for key in sums.keys():
            averages[key] = sums[key] / valid_entries_count
    else:
        print(f"  No valid entries found in {json_file_path} to calculate averages. Skipping CSV export for this file.")
        continue # Skip to the next JSON file

    # Step 2c: Determine output CSV filename
    # Create a CSV filename based on the input JSON filename
    # e.g., if input is "my_generated_data.json", output will be "my_generated_data_averages.csv"
    base_name = os.path.basename(json_file_path) # Get filename without extension
    csv_file_name = f"{base_name}_averages.csv"
    output_folder = os.path.join(script_dir, '..', "ppl_23_table")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) # Create the output folder if it doesn't exist
    csv_file_path = os.path.join(output_folder, csv_file_name) # Full path for the CSV file

    # Step 2d: Write averages to the new CSV file
    # Define the header row for the CSV
    csv_header = ["Type", "Overall Perplexity", "Chosen Perplexity", "Rejected Perplexity"]
    # Define the data rows for the CSV, using the calculated averages
    # Using .get(key, "N/A") in case a key was somehow missing from averages (though it shouldn't be with current logic)
    csv_rows = [
        ["Ref", f"{averages.get('ref_perplexity', 0):.4f}", f"{averages.get('ref_chosen_perplexity', 0):.4f}", f"{averages.get('ref_rejected_perplexity', 0):.4f}"],
        ["Policy", f"{averages.get('policy_perplexity', 0):.4f}", f"{averages.get('policy_chosen_perplexity', 0):.4f}", f"{averages.get('policy_rejected_perplexity', 0):.4f}"]
    ]

    try:
        # Open the CSV file in write mode
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile) # Create a CSV writer object
            writer.writerow(csv_header) # Write the header row
            writer.writerows(csv_rows)   # Write the data rows (Ref and Policy averages)
        print(f"  Successfully processed and exported averages to: {csv_file_path}")
    except IOError:
        print(f"  Error: Could not write to CSV file '{csv_file_path}'. Check permissions or disk space.")
    except Exception as e:
        print(f"  An unexpected error occurred while writing CSV for {json_file_path}: {e}")

print("\nAll relevant JSON files processed.")