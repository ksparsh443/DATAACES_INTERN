import pandas as pd
import requests

# Function to find the category for an article based on its content
def get_category(article, category_ids):
    for category_id, category_description in category_ids.items():
        if category_description.lower() in article.lower():
            return category_description
    return None

# URLs for the dataset and categories files
dataset_url = 'https://drive.google.com/uc?id=1mAkm62rINY_l5ZOhtp8fJGPp6nRm1jWt'
categories_url = 'https://drive.google.com/uc?id=1m3zfhMlJmQFwwza7r6Dga04wpKPljhG9'

# Download dataset and categories files using pandas
dataset = pd.read_csv(dataset_url)
categories = pd.read_csv(categories_url)

# Create a dictionary that maps category IDs to their descriptions
category_ids = {}
for index, row in categories.iterrows():
    category_id = row['id']
    category_name = row['name']
    category_description = row['description']
    category_ids[category_id] = category_description

# Initialize a list to store the output data
output_data = []

# Classify the articles and populate the output list with categorized data
for index, row in dataset.iterrows():
    article_parts = [str(row['news']), str(row['title']), str(row['snippet']), str(row['content'])]
    article = ' '.join(article_parts)
    category = get_category(article, category_ids)
    output_data.append({'Article': article, 'Category': category})  # Store category in the 'Category' column

# Create a DataFrame from the output list
output_df = pd.DataFrame(output_data)

# Save the classified dataset to a CSV file
classified_dataset_file = 'classified_dataset.csv'
output_df.to_csv(classified_dataset_file, index=False)

# Print completion message
print("Classification completed. Classified dataset saved as", classified_dataset_file)
