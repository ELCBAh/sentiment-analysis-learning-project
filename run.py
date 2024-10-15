"""
This script is used to run the sentiment analysis project.
"""

from dataset_downloader import dataset_downloader
from analizer import load_dataset, vectorize_data, train_model, evaluate_model, predict_sentiment
import os

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

    # Training the model
    model = train_model(train_vectors, train_data)

    # Evaluating the model
    accuracy = evaluate_model(model, test_vectors, test_data)
    print(f"\nModel accuracy: {accuracy:.2f}%")

    # Predicting the sentiment of a given text
    text = ["This model is working but poorly", "I love this product! It's amazing.", "I hate this product! It's terrible.", "I will probably choose a different option next time."]
    sentiment = predict_sentiment(model, vectorizer, text)
    print(f"\nThe sentiment of the text is: {sentiment}")
