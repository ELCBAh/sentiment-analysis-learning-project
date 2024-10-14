"""
This is the main structure of the project.
"""
from dataset_downloader import dataset_downloader
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

# Printing pandas data.
if __name__ == "__main__":
    # Downloading and extracting the dataset
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    filename = "aclImdb_v1.tar.gz"
    dataset_downloader(url, filename)

    # Loading the dataset
    train_data = load_dataset(os.path.join("aclImdb", "train"))
    test_data = load_dataset(os.path.join("aclImdb", "test"))

    # Printing the dataset info
    print("\nTrain data info:")
    print(train_data.info())
    print("\nTrain data:")
    print(train_data.head())
    print("\nTest data info:")
    print(test_data.info())
    print("\nTest data:")
    print(test_data.head())

    # Converting data to vectors
    train_vectors, test_vectors, vectorizer = vectorize_data(train_data, test_data)

    # Printing the vectors shape
    print("\nTrain vectors shape:", train_vectors.shape)
    print("Test vectors shape:", test_vectors.shape)
