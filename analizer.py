"""
This is the main structure of the project.
"""

import os
import urllib.request
import tarfile
import pandas as pd
import numpy as np

# Defining dataset Url

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filename = "aclImdb_v1.tar.gz"


# Downloading and extracting the dataset

# Uncomment the following lines to download and extract the dataset
"""

if not os.path.exists(filename):
    print("Downloading the dataset...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete.")

if not os.path.exists("aclImdb"):
    print("Extracting the dataset...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()
    print("Extraction complete.")

"""

# Loading the dataset using pandas

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

# Printing pandas data.

train_data = load_dataset(os.path.join("aclImdb", "train"))
test_data = load_dataset(os.path.join("aclImdb", "test"))
print("\nTrain data info:")
print(train_data.info())
print("\nTrain data:")
print(train_data.head())
print("\nTest data info:")
print(test_data.info())
print("\nTest data:")
print(test_data.head())
