import pandas as pd

def get_category(article, category_ids):
    for category_id, category_description in category_ids.items():
        if category_description.lower() in article.lower():
            return category_description
    return None

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
    category = get_category(article, category_ids)
    output_data.append({'Article': article, 'Category': category})

    print("Article:", article)
    print("Category:", category_ids.get(category))
    print("-" * 40)

output_df = pd.DataFrame(output_data)

classified_dataset_file = 'classified_dataset.csv'
output_df.to_csv(classified_dataset_file, index=False)

print("Classification completed. Classified dataset saved as", classified_dataset_file)
