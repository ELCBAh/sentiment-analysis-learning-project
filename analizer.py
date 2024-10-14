"""
This is the main structure of the project.
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

def load_dataset(data_dir):
    """Function to load the dataset into a pandas DataFrame.

    Args:
        data_dir (string): path to the dataset

    Returns:
        pandas DataFrame: Returns a dataframe with the review and sentiment columns.
    """
    data = []
    for sentiment in ["pos", "neg"]:
        path = os.path.join(data_dir, sentiment)
        if not os.path.exists(path):
            print(f"Path {path} does not exist")
            continue
        file_count = len(os.listdir(path))
        for i, file in enumerate(os.listdir(path), 1):
            if i % 1000 == 0:
                print(f"Processed {i}/{file_count} files for sentiment {sentiment}")
            try:
                with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                    text = f.read()
                    data.append((text, 1 if sentiment == "pos" else 0))
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError for file {file}. Skipping this file.")
            except Exception as e:
                print(f"Error reading file {file}: {e}")
    return pd.DataFrame(data, columns=["review", "sentiment"])

# Converting data to vectors

def vectorize_data(train_data, test_data):
    """
    Convert text data into TF-IDF vectors.

    Args:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Test data

    Returns:
        tuple: (train_vectors, test_vectors, vectorizer)
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    train_vectors = vectorizer.fit_transform(train_data['review'])
    test_vectors = vectorizer.transform(test_data['review'])
    return train_vectors, test_vectors, vectorizer
