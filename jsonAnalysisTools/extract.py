import json
import os
import glob
import sys

def extract_data(json_path, store_path):
    """
    Extract specific fields from a JSON list of QA items and save the result.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract required fields
    extracted = []
    for item in data:
        extracted.append({
            "question": item.get("question", ""),
            "ref_response": item.get("ref_response", ""),
            "policy_response": item.get("policy_response", ""),
            "expected": item.get("expected", "")
        })

    # Save with same filename to store_path
    filename = os.path.basename(json_path)
    output_path = os.path.join(store_path, filename)
    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump(extracted, out_file, indent=2, ensure_ascii=False)


def main():
    """
    Find all JSON files under the 'results' folder whose filenames contain
    'generated', extract specific fields, and save each to a new directory.
    """
    results_dir = '../results'
    store_path = '../extractedData'

    file_list = glob.glob(os.path.join(results_dir, '**', '*generated*.json'), recursive=True)
    if not file_list:
        print("No JSON files found containing 'generated' in their names.")
        sys.exit(1)

    print(f"Found {len(file_list)} JSON files to process.")

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    for json_path in file_list:
        extract_data(json_path, store_path)
        print(f"Processed {json_path} and saved to {store_path}")


if __name__ == '__main__':
    main()
