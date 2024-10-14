"""
Defining dataset Url
"""

import urllib.request
import tarfile
import os

def dataset_downloader(url, filename):
	if not os.path.exists(filename):
		print("Downloading the dataset...")
		urllib.request.urlretrieve(url, filename)
		print("Download complete.")

	if not os.path.exists("aclImdb"):
		print("Extracting the dataset...")
		with tarfile.open(filename, "r:gz") as tar:
			tar.extractall()
		print("Extraction complete.")

if __name__ == "__main__":
    dataset_downloader()
