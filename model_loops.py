from utils import (
    checkpoint_verification,
    create_data_loader,
    create_train_test_split,
    load_base_model,
    load_fine_tuned_model,
    train_one_step
)
import torch
from torch.optim import AdamW
from collections import defaultdict
from tqdm import tqdm
from analysis import get_word_attribution, model_evaluation, train_with_explaination_one_step
import pandas as pd
import json
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import os

def fine_tune_loop(train_df, base_model="google-bert/bert-base-cased", fine_tuned_model_path="model_weights/test_fine_tuned_bert.pth", 
                   epochs=3, batch_size=16, learning_rate=2e-5, bullshit_words=None):
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
    train_loader = create_data_loader(
        train_df,
        tokenizer,
        batch_size=batch_size,
        bullshit_words=bullshit_words
    )


    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate) # Learning rate can be adjusted

    # Training loop
    for epoch in range(epochs):

        avg_loss = train_one_step(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save the fine-tuned model
    torch.save(model.state_dict(), fine_tuned_model_path)

    return fine_tuned_model_path

def fine_tune_with_explanaitions(train_df, 
                                 base_model="google-bert/bert-base-cased", 
                                 fine_tuned_model_path="model_weights/fine_tuned_bert_with_ex.pth", 
                                 epochs=3, 
                                 batch_size=16, 
                                 learning_rate=2e-5,
                                 bullshit_words=[],
                                 explanaition_loss_ratio=0.2,
                                 lam=1.0,
                                 n_steps=500,
                                 checkpoint_every_n_step = 1
                                ):
    """fine tunes the model, using the regular loss alongside the loss of the explanaition method 

    Args:
        train_df (Pandas.Dataset): training set
        base_model (str, optional): path to the HuggingFace mode which is used. Defaults to "google-bert/bert-base-cased".
        fine_tuned_model_path (str, optional): _description_. Defaults to "fine_tuned_bert_with_ex.pth".
        epochs (int, optional): _description_. Defaults to 3.
        batch_size (int, optional): _description_. Defaults to 16.
        learning_rate (_type_, optional): _description_. Defaults to 2e-5.
        bullshit_words (list, optional): _description_. Defaults to [].
        explanaition_loss_ratio (float, optional): % wise how much the dataset will be trained with explanaitions _description_. Defaults to 0.2.
        lam (float, optional): lambda value which is multiplied with the IG loss to make it's influece more or less. Defaults to 1.0.
        n_steps (int, optional): how many steps for IG. Defaults to 500.
        checkpoint_every_n_step (int, optional): creates a checkpoint every n'th batch computed in the second learning phase. Defaults to 1.

    Returns:
        String: path to the (final) fine tunes weights
    """
    
    checkpoint_path = "model_weights/checkpoint.pth"
    
    tokenizer, model, device = load_base_model(base_model)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

   #Converts the string tokens to BERT compatible input ids 
    bullshit_token_ids = set()
    for word in bullshit_words:
        # encode without special tokens [CLS]/[SEP] to get the raw IDs
        ids = tokenizer.encode(word, add_special_tokens=False)
        bullshit_token_ids.update(ids)

    # Convert to a tensor for use with torch.isin inside the loss function
    bullshit_ids_tensor = torch.tensor(list(bullshit_token_ids)).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"reload data from last failed run")
    else:
        start_epoch = 0
    
    for epoch in range(start_epoch, epochs):
        first_phase_avg_loss, second_phase_avg_loss = train_with_explaination_one_step(
            model=model,
            tokenizer=tokenizer,
            train_df=train_df,
            optimizer=optimizer,
            device=device,
            bullshit_ids=bullshit_ids_tensor,
            explanaition_loss_ratio=explanaition_loss_ratio,
            lam=lam,
            n_steps=n_steps,
            batch_size=batch_size,
            checkpoint_every_n_step=checkpoint_every_n_step,
            checkpoint_path=checkpoint_path,
            epoch=epoch,
        )
        
        print(f"Epoch {epoch + 1}/{epochs}, First Phase Average Loss: {first_phase_avg_loss:.4f}, Second Phase Average Loss: {second_phase_avg_loss:.4f}")
    
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

    reviews = train_df[review_column].tolist()

    print(f"Amount of reviews: {len(reviews)}")

    # Check for existing checkpoint and get all neccessary data to resume or start fresh
    start_index, word_sums, word_counts, total_abs_delta = checkpoint_verification(checkpoint_json)

    # If already finished, just return
    if start_index >= len(reviews):
        print("Calcualtion was already completed in previous run.")
        return output_json, total_abs_delta / len(reviews)
    
    # Start from the last saved index
    reviews_to_process = reviews[start_index:]
    
    # Calculate word attributions
    for i, review in enumerate(tqdm(reviews_to_process, desc="Calculating Attributions", initial=start_index, total=len(reviews))):

        # Get the index in the original reviews list
        current_real_index = start_index + i

        word_attr, delta = get_word_attribution(n_steps, review, model, tokenizer, internal_batch_size=internal_batch_size)

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

def extract_logits(train_df, output_json="review_logits.json", review_column="review", batch_size=32):
    """
    Extracts logits from the fine-tuned model for all reviews in the provided dataframe.
    Saves the logits to a JSON file.

    :param train_df: DataFrame containing training data.
    :param output_json: Path to save the output JSON file with logits.
    :param review_column: Name of the column containing the reviews.
    :param batch_size: Batch size for processing reviews.
    """
    
    tokenizer, model, device = load_fine_tuned_model()
    
    model.to(device)
    model.eval()

    reviews = train_df[review_column].tolist()
    logits_results = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(reviews), batch_size)):

            # Create batch
            batch_texts = reviews[i:i + batch_size]
            
            # Tokenize and encode the batch
            encoded = tokenizer(
                batch_texts, 
                truncation=True, 
                padding=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(device)

            # Get model outputs
            outputs = model(**encoded)
            batch_logits = outputs.logits.cpu().numpy()

            # Save logits for each review
            for j, logit in enumerate(batch_logits):
                
                logits_results.append({
                    "review_index": i + j,
                    "logit_0": float(logit[0]), # Negative class logit
                    "logit_1": float(logit[1]), # Positive class logit
                    "prediction": int(logit.argmax()) # Predicted class
                })

    with open(output_json, "w") as f:
        json.dump(logits_results, f, indent=4)

if __name__ == "__main__":
    train_df, test_df = create_train_test_split()
    # extract_logits(pth_model_path="fine_tuned_bert.pth", bert_base_name="google-bert/bert-base-cased", train_df=train_df)
    bullshit_words = [
        "Zero", "Rather", "turkey", "sheer", "hi", "episodes", "episode", "today", "flu",
        "summary", "sports", "Kid", "block", "Sinatra", "born", "Anyone", "entry", "Although", "Bourne",
        "app", "DVD", "short", "animated", "Yet", "Many", "Not", "scenery", "beginning", "Day", "bit",
        "adult", "describe", "true", "personally", "ready", "match"
    ]