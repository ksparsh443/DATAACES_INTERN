import openai
import pandas as pd

openai.api_key = 'test123'

def classify_text(text, category_ids):
    prompt = f"Classify the following text into categories: {category_ids}. Text: {text}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0.7
    )
    category = response.choices[0].text.strip()
    return category

dataset_url = 'https://drive.google.com/uc?id=1mAkm62rINY_l5ZOhtp8fJGPp6nRm1jWt'
categories_url = 'https://drive.google.com/uc?id=1m3zfhMlJmQFwwza7r6Dga04wpKPljhG9'

dataset = pd.read_csv(dataset_url)
categories = pd.read_csv(categories_url)

category_ids = {}
for index, row in categories.iterrows():
    category_id = row['id']
    category_description = row['description']
    category_ids[category_id] = category_description

output_data = []

for index, row in dataset.iterrows():
    article_parts = [str(row['news']), str(row['title']), str(row['snippet']), str(row['content'])]
    article = ' '.join(article_parts)
    category = classify_text(article, category_ids)
    output_data.append({'Article': article, 'Category': category})

    print("Article:", article)
    print("Category:", category)
    print("-" * 40)

output_df = pd.DataFrame(output_data)

classified_dataset_file = 'classified_dataset.csv'
output_df.to_csv(classified_dataset_file, index=False)

print("Classification completed. Classified dataset saved as", classified_dataset_file)
