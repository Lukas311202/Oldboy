from utils import checkpoint_verification, create_data_loader, create_train_test_split, load_base_model, load_fine_tuned_model, train_one_step
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
    Calculates word attributions for all reviews in the provided dataframe using the fine-tuned model.
    Saves complete progress with all neccessary data to continue at a later point.
    If incomplete progress is detected, the savestate is loaded and calculation continues from there.

    :param train_df: DataFrame containing training data.
    :param n_steps: Number of steps for Integrated Gradients.
    :param save_every: Save progress every N reviews.
    :param fine_tuned_model_path: Path to the fine-tuned model.
    :param output_json: Path to save the output JSON file with word attributions.
    :param review_column: Name of the column containing the reviews.

    :return: Path to the output JSON file.
    :return: final_avg_delta: Average delta value across all reviews.
    """
           
    checkpoint_json =  output_json.replace(".json", "_stats.json")

    # Load reviews
    reviews = train_df[review_column].tolist()

    print(f"Amount of reviews: {len(reviews)}")

    # Check for existing checkpoint and get all neccessary data to resume or start fresh
    start_index, word_sums, word_counts, total_abs_delta = checkpoint_verification(checkpoint_json)

    # If already finished, just return
    if start_index >= len(reviews):
        print("Calcualtion was already completed in previous run. Loading results.")
        return output_json, total_abs_delta / len(reviews)
    
    # Start from the last saved index
    reviews_to_process = reviews[start_index:]
    
    # Calculate word attributions
    for i, review in enumerate(tqdm(reviews_to_process, desc="Calculating Attributions", initial=start_index, total=len(reviews))):

        # Get the index in the original reviews list
        current_real_index = start_index + i

        word_attr, delta = get_word_attribution(n_steps, review, model, tokenizer)

        total_abs_delta += abs(delta)

        for word, value in word_attr.items():
            word_sums[word] += value
            word_counts[word] += 1

        # Checkpoint
        if (i + 1) % save_every == 0:

            checkpoint_data = {
                "reviews_processed": current_real_index + 1,
                "total_abs_delta": total_abs_delta,
                "word_sums": word_sums,   
                "word_counts": word_counts 
            }

            current_avg = {word: word_sums[word] / word_counts[word] for word in word_sums}

            # Save checkpoint results periodically
            with open(checkpoint_json, "w") as f:
                json.dump(checkpoint_data, f, indent=4)

            # Save current average attributions
            with open(output_json, "w") as f:
                json.dump(current_avg, f, indent=4)

            print(f"Processed {current_real_index + 1} reviews, last delta: {delta}, last average delta: {total_abs_delta / (current_real_index + 1)}")

    # Calculate averages of attributions
    word_avg = {word: word_sums[word] / word_counts[word] for word in word_sums}

    # Save IG results to provided JSON path
    with open(output_json, "w") as f:
        json.dump(word_avg, f, indent=4)

    # Get final average delta
    final_avg_delta = total_abs_delta / len(reviews)

    # Save final stats
    final_stats = {
        "reviews_processed": len(reviews),
        "total_abs_delta": total_abs_delta,
        "final_avg_delta": final_avg_delta,   
        "status": "completed",
        "word_sums": word_sums, 
        "word_counts": word_counts
    }

    with open(checkpoint_json, "w") as f:
        json.dump(final_stats, f, indent=4)

    return output_json, final_avg_delta
