from torch.optim import AdamW
from utils import create_train_test_split, load_model, train_one_step, create_data_loader
import torch

if __name__ == "__main__":

    # Load model and tokenizer
    tokenizer, model, device = load_model("google-bert/bert-base-cased")

    # Create train-test split and data loader
    train_df, test_df = create_train_test_split(data="imdb_dataset.csv", text_column="review", label_column="sentiment",
                                                test_size=0.2, seed=42, stratify=True)
    train_loader = create_data_loader(train_df, tokenizer, batch_size=16)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5) # Learning rate can be adjusted

    epochs = 3
    # Training loop
    for epoch in range(epochs):

        avg_loss = train_one_step(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "fine_tuned_bert.pth")