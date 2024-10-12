"""
This is the main structure of the project.
"""

import os
import urllib.request
import tarfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import re

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

def load_dataset():
    pos_dir = "aclImdb/train/pos"
    neg_dir = "aclImdb/train/neg"
    data = []

    for file in os.listdir(pos_dir):
        with open(os.path.join(pos_dir, file), "r", encoding="utf-8") as f:
            data.append((f.read(), 1))
    for file in os.listdir(neg_dir):
        with open(os.path.join(neg_dir, file), "r", encoding="utf-8") as f:
            data.append((f.read(), 0))
    return data

data = load_dataset()
print(len(data))
print(data[0])
