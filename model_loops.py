from utils import checkpoint_verification, create_data_loader, create_train_test_split, load_base_model, load_fine_tuned_model, train_one_step
import torch
from torch.optim import AdamW
from collections import defaultdict
from tqdm import tqdm
from analysis import get_word_attribution
import pandas as pd
import json
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
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

def run_attributions(n_steps, save_every, internal_batch_size, tokenizer, model, train_df, output_json="global_word_attributions.json", review_column="review"):
    """
    Calculates word attributions for all reviews in the provided dataframe using the fine-tuned model.
    Saves complete progress with all neccessary data to continue at a later point.
    If incomplete progress is detected, the savestate is loaded and calculation continues from there.

    :param train_df: DataFrame containing training data.
    :param n_steps: Number of steps for Integrated Gradients.
    :param save_every: Save progress every N reviews.
    :param internal_batch_size: Batch size for internal model processing.
    :param model: Fine-tuned model for attribution calculation.
    :param tokenizer: Tokenizer corresponding to the model.
    :param fine_tuned_model_path: Path to the fine-tuned model.
    :param output_json: Path to save the output JSON file with word attributions.
    :param review_column: Name of the column containing the reviews.

    :return: Path to the output JSON file.
    :return: final_avg_delta: Average delta value across all reviews.
    """
           
    checkpoint_json =  output_json.replace(".json", "_stats.json")

    # Load reviews
    reviews = train_df[review_column].tolist()

    #reviews = create_data_loader(train_df, tokenizer, batch_size=4)

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

        word_attr, delta = get_word_attribution(n_steps, review, model, tokenizer, internal_batch_size=internal_batch_size)

        # Convert delta to float
        delta_float = delta.item() if torch.is_tensor(delta) else delta

        total_abs_delta += abs(delta_float)

        # Extract tokens and attributions
        tokens_list = word_attr["tokens"][0]  # Get first review's tokens
        attributions = word_attr["attributions"]  # Attribution scores for each token
        
        # Zip tokens with their attribution values and aggregate
        for token, attribution_value in zip(tokens_list, attributions):
            # Convert tensor to float if necessary
            attr_val = attribution_value.item() if torch.is_tensor(attribution_value) else attribution_value
            word_sums[token] += attr_val
            word_counts[token] += 1

        # Checkpoint
        if (i + 1) % save_every == 0:

            # Convert tensors to floats for JSON serialization
            word_sums_float = {word: float(word_sums[word]) for word in word_sums}
            
            checkpoint_data = {
                "reviews_processed": current_real_index + 1,
                "total_abs_delta": float(total_abs_delta),
                "word_sums": word_sums_float,   
                "word_counts": word_counts 
            }

            current_avg = {word: word_sums_float[word] / word_counts[word] for word in word_sums_float}

            # Save checkpoint results periodically
            with open(checkpoint_json, "w") as f:
                json.dump(checkpoint_data, f, indent=4)

            # Save current average attributions
            with open(output_json, "w") as f:
                json.dump(current_avg, f, indent=4)

            print(f"Processed {current_real_index + 1} reviews, last delta: {delta_float}, last average delta: {total_abs_delta / (current_real_index + 1)}")

    # Calculate averages of attributions
    word_avg = {word: word_sums[word] / word_counts[word] for word in word_sums}

    # Save IG results to provided JSON path
    with open(output_json, "w") as f:
        json.dump(word_avg, f, indent=4)

    # Get final average delta
    final_avg_delta = total_abs_delta / len(reviews)

    # Save final stats
    word_sums_float = {word: float(word_sums[word]) for word in word_sums}
    final_stats = {
        "reviews_processed": len(reviews),
        "total_abs_delta": float(total_abs_delta),
        "final_avg_delta": float(final_avg_delta),   
        "status": "completed",
        "word_sums": word_sums_float, 
        "word_counts": word_counts
    }

    with open(checkpoint_json, "w") as f:
        json.dump(final_stats, f, indent=4)

    return output_json, final_avg_delta

def extract_logits(pth_model_path, bert_base_name, train_df, output_json="review_logits.json", review_column="review", batch_size=32):

    #TODO: REFACTOR THIS FUNCTION

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Initialisiere Modell-Architektur ({bert_base_name})...")
    tokenizer = BertTokenizer.from_pretrained(bert_base_name)
    
    # 1. Architektur laden (muss exakt zum Training passen, z.B. num_labels=2)
    model = BertForSequenceClassification.from_pretrained(bert_base_name, num_labels=2)
    
    # 2. Die gespeicherten Gewichte aus der .pth Datei laden
    print(f"Lade Gewichte aus {pth_model_path}...")
    state_dict = torch.load(pth_model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    reviews = train_df[review_column].tolist()[:100]  # Nur die ersten 100 Reviews für dieses Beispiel
    logits_results = []

    print(f"Berechne Logits für {len(reviews)} Reviews...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(reviews), batch_size)):
            batch_texts = reviews[i:i + batch_size]
            
            encoded = tokenizer(
                batch_texts, 
                truncation=True, 
                padding=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(device)

            outputs = model(**encoded)
            batch_logits = outputs.logits.cpu().numpy()

            for j, logit in enumerate(batch_logits):
                # Wir speichern beide Logits und die finale Entscheidung
                logits_results.append({
                    "review_index": i + j,
                    "logit_0": float(logit[0]), # Score für Negative
                    "logit_1": float(logit[1]), # Score für Positive
                    "prediction": int(logit.argmax())
                })

    # Speichern als JSON
    with open(output_json, "w") as f:
        json.dump(logits_results, f, indent=4)
    
    print(f"✅ Fertig! Logits gespeichert in {output_json}")

if __name__ == "__main__":
    train_df, test_df = create_train_test_split()
    extract_logits(pth_model_path="fine_tuned_bert.pth", bert_base_name="google-bert/bert-base-cased", train_df=train_df)