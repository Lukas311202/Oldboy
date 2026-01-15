import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer,  AutoModelForSequenceClassification

def load_base_model(model_name="google-bert/bert-base-cased") -> tuple[AutoTokenizer, torch.nn.Module, any]:
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

def load_weighted_model(model_name="google-bert/bert-base-cased", model_path="fine_tuned_bert.pth"):

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

    # Set model into evaluation mode
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

    for batch in data_load:
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

def create_data_loader(df, tokenizer, batch_size=16, max_length=128):

    # Tokenize the texts
    input_encodings = tokenizer(
        df['review'].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    labels = torch.tensor(df['sentiment'].tolist())

    # Create Tensordataset instance
    dataset = TensorDataset(
        input_encodings['input_ids'],
        input_encodings['attention_mask'],
        labels
    )

    # Create Dataloader instance
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader