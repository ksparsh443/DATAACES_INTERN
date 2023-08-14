import pandas as pd
import requests
import numpy as np
import tensorflow as tf

def get_category(article, category_ids):
    # Convert the article to a sequence of integers
    article_tokens = article.split()
    article_ints = [get_word_index(token) for token in article_tokens]

    # Pad the sequence to the maximum length
    max_seq_len = 100
    padded_article = pad_sequences([article_ints], maxlen=max_seq_len, padding='post')

    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(word_index) + 1, 128),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(category_ids), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(padded_articles, category_labels, epochs=10)

    # Predict the category of the article
    prediction = model.predict(padded_article)
    category_id = np.argmax(prediction)
    category = category_ids[category_id]

    return category

# File URLs
dataset_url = 'https://drive.google.com/uc?id=1mAkm62rINY_l5ZOhtp8fJGPp6nRm1jWt'
categories_url = 'https://drive.google.com/uc?id=1m3zfhMlJmQFwwza7r6Dga04wpKPljhG9'

# Download dataset and categories files
dataset = pd.read_csv(dataset_url)
categories = pd.read_csv(categories_url)

# Create a dictionary that maps category IDs to descriptions
category_ids = {}
for index, row in categories.iterrows():
    category_id = row['id']
    category_name = row['name']
    category_description = row['description']
    category_ids[category_id] = category_description

# Create a mapping from words to integers
word_index = {}
i = 0
for article in dataset['article']:
    for token in article.split():
        if token not in word_index:
            word_index[token] = i
            i += 1

# Create the padded articles and category labels
padded_articles = pad_sequences(dataset['Article'].apply(lambda x: [word_index[token] for token in x.split()]),
                                maxlen=100, padding='post')
category_labels = tf.keras.utils.to_categorical(dataset['Category'])

# Train the model
model.fit(padded_articles, category_labels, epochs=10)

# Classify the articles
for index, row in dataset.iterrows():
    article = row['Article']
    category = get_category(article, category_ids)
    print('Article:', article, 'Category:', category)
