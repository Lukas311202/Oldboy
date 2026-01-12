import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,  AutoModelForSequenceClassification

def load_model(model_name="google-bert/bert-base-cased"):
    """
    Load a pre-trained model and tokenizer for binary text classification.
    
    :param model_name: The name of the pre-trained model to load.
    """

    # Load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with binary classification head
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Configuration of binary classification labels
    model.config.id2label = {0: "negative", 1: "positive"}
    model.config.label2id = {"negative": 0, "positive": 1}

    # Evaluation mode
    model.eval()

    return tokenizer, model, device

def create_train_test_split(data="imdb_dataset.csv", text_column="review", label_column="sentiment", 
                      test_size=0.2, seed=42, stratify=True):
    """
    Splits the dataset into training and testing sets.

    :param data: Path to the dataset file.
    :param text_column: Name of the column containing the text data.
    :param label_column: Name of the column containing the label data.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_seed: Random seed for reproducibility.
    :param stratify: Whether to stratify the split based on labels.
    """

    # Read data
    df = pd.read_csv(data)

    # Convert labels to binary
    df[label_column] = df[label_column].map({"positive": 1, "negative": 0})

    # Split data
    stratify_labels = df[label_column] if stratify else None
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify_labels)

    print(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")

    return train_df, test_df


