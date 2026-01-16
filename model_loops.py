from utils import create_data_loader, create_train_test_split, load_base_model, load_fine_tuned_model, train_one_step
import torch
from torch.optim import AdamW
from collections import defaultdict
from tqdm import tqdm
from analysis import get_word_attribution
import pandas as pd
import json

def fine_tune_loop(train_df, base_model="google-bert/bert-base-cased", fine_tuned_model_path="fine_tuned_bert.pth", 
                   epochs=3, batch_size=16, learning_rate=2e-5):
    """
    Fine-tunes the BERT model on the IMDB dataset. Saves the output model to the specified path.
    
    :param train_df: DataFrame containing training data.
    :param base_model: Pre-trained model name.
    :param fine_tuned_model_path: Path to save the fine-tuned model.
    :param epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :param learning_rate: Learning rate for the optimizer.

    :return: Path to the fine-tuned model.
    """

    # Load model and tokenizer
    tokenizer, model, device = load_base_model(base_model)

    # Create data loader
    train_loader = create_data_loader(train_df, tokenizer, batch_size=batch_size)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate) # Learning rate can be adjusted

    # Training loop
    for epoch in range(epochs):

        avg_loss = train_one_step(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save the fine-tuned model
    torch.save(model.state_dict(), fine_tuned_model_path)

    return fine_tuned_model_path

def run_attributions(n_steps, save_every, tokenizer, model, train_df, output_json="global_word_attributions.json", review_column="review"):
    """
    Caclulates word attributions for all reviews in the provided dataframe using the fine-tuned model.

    :param train_df: DataFrame containing training data.
    :param n_steps: Number of steps for Integrated Gradients.
    :param save_every: Save progress every N reviews.
    :param fine_tuned_model_path: Path to the fine-tuned model.
    :param output_json: Path to save the output JSON file with word attributions.
    :param review_column: Name of the column containing the reviews.
*
    :return: Path to the output JSON file.
    :return: final_avg_delta: Average delta value across all reviews.
    """
           
    # Load reviews
    reviews = train_df[review_column].tolist()

    print(f"Amount of reviews: {len(reviews)}")

    # Initialize accumulators
    word_sums = defaultdict(float)
    word_counts = defaultdict(int)

    # Path for delta stats
    stats_json_path = output_json.replace(".json", "_stats.json")

    total_abs_delta = 0.0

    # Calculate word attributions
    for i, review in enumerate(tqdm(reviews, desc="Calculating Attributions", unit="review")):
        word_attr, delta = get_word_attribution(n_steps, review, model, tokenizer)

        total_abs_delta += abs(delta)

        for word, value in word_attr.items():
            word_sums[word] += value
            word_counts[word] += 1

        if (i + 1) % save_every == 0:
            current_avg = {word: (word_sums[word] / word_counts[word]).item() for word in word_sums}

            # Save IG resulsts periodically
            with open(output_json, "w") as f:
                json.dump(current_avg, f, indent=4)

            # Save delta stats periodically
            current_avg_delta = total_abs_delta / (i + 1)
            stats_data = {
                "reviews_processed": i + 1,
                "current_avg_abs_delta": current_avg_delta,
                "last_single_delta": delta
            }
            with open(stats_json_path, "w") as f:
                json.dump(stats_data, f, indent=4)

            print(f"Processed {i + 1} reviews, last delta: {delta}")

    # Calculate averages of attributions
    word_avg = {word: (word_sums[word] / word_counts[word]).item() for word in word_sums}

    # Get final average delta
    final_avg_delta = total_abs_delta / len(reviews)

    # Save JSON to provided path
    with open(output_json, "w") as f:
        json.dump(word_avg, f, indent=4)

    # Save delta stats
    final_stats = {
        "reviews_processed": len(reviews),
        "final_avg_abs_delta": final_avg_delta,
        "status": "completed"
    }
    with open(stats_json_path, "w") as f:
        json.dump(final_stats, f, indent=4)

    return output_json, final_avg_delta
