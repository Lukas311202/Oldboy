from collections import defaultdict
import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer,  AutoModelForSequenceClassification
import os
from tqdm import tqdm
import re

def load_base_model(model_name="google-bert/bert-base-cased"):
    """
    Load a pre-trained model and tokenizer for binary text classification.
    
    :param model_name: The name of the pre-trained model to load.

    :return: tokenizer, model, device
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

def load_fine_tuned_model(model_name="google-bert/bert-base-cased", model_path="model_weights/fine_tuned_bert.pth"):
    """
    Loads a fine-tuned model from given path. 

    :param model_name: The name of the pre-trained model to load.
    :param model_path: Path to the fine-tuned model weights.

    :return: tokenizer, model, device
    """

    # Load base model
    tokenizer, model, device = load_base_model(model_name)

    # Load fine-tuned weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    return tokenizer, model, device


def create_train_test_split(data="imdb_dataset.csv", label_column="sentiment", 
                      test_size=0.2, seed=42, stratify=True):
    """
    Splits the dataset into training and testing sets.

    :param data: Path to the dataset file.
    :param label_column: Name of the column containing the label data.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_seed: Random seed for reproducibility.
    :param stratify: Whether to stratify the split based on labels.

    :return: train_df, test_df
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

def mask_bullshit_words(text: str, bullshit_words: list, mask_token: str):
    """
    Replaces bullshit words in text with [MASK], word-boundary aware.

    :param text: Original text.
    :param bullshit_words: List of words to mask.
    :param mask_token: Token to replace bullshit words with.

    :return: Masked text.
    """
    for word in bullshit_words:
        pattern = r"\b" + re.escape(word) + r"\b"
        text = re.sub(pattern, mask_token, text)
    return text

def train_one_step(model, data_load, optimizer, device):
    """
    Trains the model for one epoch.

    :param model: Model to train-
    :param data_load: DataLoader for training data.
    :param optimizer: The optimizer to use for training.
    :param device: Either 'cpu' or 'cuda'.

    :return: Average loss for the epoch.
    """
    # Set model to training mode
    model.train()
    total_loss = 0

    for batch in tqdm(data_load):
        # Move batch to device if possible
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_load)

    return avg_loss

def create_data_loader(df, tokenizer, batch_size=16, max_length=128, bullshit_words=None):
    """
    Creates a DataLoader from the given DataFrame.

    :param df: DataFrame containing the data.
    :param tokenizer: Tokenizer to use for encoding the text.
    :param batch_size: Batch size for the DataLoader.
    :param max_length: Maximum sequence length for tokenization.
    :param bullshit_words: List of words to mask in the text.

    :return: DataLoader
    """

    texts = df['review'].tolist()

    if bullshit_words is not None and len(bullshit_words) > 0:
        texts = [
            mask_bullshit_words(
                text=text,
                bullshit_words=bullshit_words,
                mask_token=tokenizer.mask_token
            )
            for text in texts
        ]

    input_encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    labels = torch.tensor(df['sentiment'].tolist())

    dataset = TensorDataset(
        input_encodings['input_ids'],
        input_encodings['attention_mask'],
        labels
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def checkpoint_verification(checkpoint_json):
    """
    Loads checkpoint data for resuming attribution calculations if available.
    Else returns initial values.

    :param checkpoint_json: Path to the checkpoint JSON file.

    :return: start_index, word_sums, word_counts, total_abs_delta
    """

    # Initialize accumulators
    start_index = 0
    word_sums = defaultdict(float)
    word_counts = defaultdict(int)
    total_abs_delta = 0.0

    if os.path.exists(checkpoint_json):
        print(f"Resuming from checkpoint: {checkpoint_json}")

        try:
            with open(checkpoint_json, "r") as f:
                checkpoint_data = json.load(f)

            # Extract saved state
            start_index = checkpoint_data["reviews_processed"]
            total_abs_delta = checkpoint_data["total_abs_delta"]
            
            # Load word sums and counts
            word_sums.update(checkpoint_data["word_sums"])
            word_counts.update(checkpoint_data["word_counts"])

            print(f"Resumed from review index: {start_index}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
    else:
        print("No checkpoint found. Starting from scratch.")

    return start_index, word_sums, word_counts, total_abs_delta


