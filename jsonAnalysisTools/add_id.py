import json
import os
from collections import OrderedDict
input_folder = '../extractedData/'
output_folder = '../id/'
output_suffix = 'id_'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

if not json_files:
    print("Can't find any JSON files.")
else:
    for file_name in json_files:
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(
            output_folder, f'{output_suffix}{file_name}'
        )

        # access the input json
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # process each item, adding question_id and placing it at the front
        new_data = []
        for idx, item in enumerate(data, start=1):
            ordered_item = OrderedDict()
            ordered_item['question_id'] = idx
            for key, value in item.items():
                ordered_item[key] = value
            new_data.append(ordered_item)

        # write to new file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Processed: {file_name} → {output_path}")