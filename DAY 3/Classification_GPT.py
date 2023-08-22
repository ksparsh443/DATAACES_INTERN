import pandas as pd
import openai
import time
import json
import jsonlines

api_key = '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'
openai.api_key = api_key


def request_classification(jsonl_file):
    classes = []
    with open(jsonl_file, 'r') as f:
        with jsonlines.Reader(f) as reader:
            for article in reader:
                # Check if the `prompt` key is present
                if 'prompt' not in article:
                    raise ValueError('The `prompt` key is missing.')

                # Add a `prompt` key to the JSON object
                article['prompt'] = article['text']
                del article['text']
                for key in article:
                    if not isinstance(article[key], str):
                        raise ValueError(f'{key} should be a string.')
                query = article['prompt']
                try:
                    query = openai.Classification.create(
                        file=jsonl_file,
                        query=query,
                        search_model="text-curie-001",
                        model="text-curie-001")
                    classes.append(query['label'])
                except Exception as e:
                    # Sleep for 60 seconds to avoid exceeding the API rate limit.
                    time.sleep(60)
    return classes


dataset_url = 'https://drive.google.com/uc?id=1mAkm62rINY_l5ZOhtp8fJGPp6nRm1jWt'
categories_url = 'https://drive.google.com/uc?id=1m3zfhMlJmQFwwza7r6Dga04wpKPljhG9'

dataset_jsonl_file = 'dataset.jsonl'
categories_jsonl_file = 'categories.jsonl'

category_ids = {}
with open(categories_jsonl_file, 'r') as f:
    with jsonlines.Reader(f) as reader:
        for row in reader:
            category_ids[row['id']] = row['description']

data = []
with open(dataset_jsonl_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

output_data = []

file_info = openai.File.create(
    file=open(dataset_jsonl_file, 'rb'), purpose="fine-tune"
)
file_id = file_info['id']




classes = request_classification(dataset_jsonl_file)

# Add the class labels to the output data
for i in range(len(data)):
    output_data.append({'Article': data[i]['text'], 'Category': classes[i]})


classified_dataset_file = 'classified_dataset.csv'
output_df = pd.DataFrame(output_data)
output_df.to_csv(classified_dataset_file, index=False)

print("Classification completed. Classified dataset saved as", classified_dataset_file)
