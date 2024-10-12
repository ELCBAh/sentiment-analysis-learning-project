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

if not os.path.exists(filename):
    print("Downloading the dataset...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete.")

if not os.path.exists("aclImdb"):
    print("Extracting the dataset...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()
    print("Extraction complete.")

# Loading the dataset using pandas

def load_dataset(data_dir):
    data = []
    for sentiment in ["pos", "neg"]:
        path = os.path.join(data_dir, sentiment)
        for file in os.listdir(path):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                text = f.read()
                data.append((text, 1 if sentiment == "pos" else 0))
    return pd.DataFrame(data, columns=["review", "sentiment"])

# Find a way to print the first row of the dataset in pandas

train_data = load_dataset("aclImdb/train")
test_data = load_dataset("aclImdb/test")
print("\nTrain data info:")
print(train_data.info())
print("\nTrain data:")
print(train_data.head())
print("\nTest data info:")
print(test_data.info())
print("\nTest data:")
print(test_data.head())
