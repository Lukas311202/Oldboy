from utils import (
    checkpoint_verification,
    create_data_loader,
    load_base_model,
    load_fine_tuned_model,
    train_one_step
)
import torch
from torch.optim import AdamW
from tqdm import tqdm
from analysis import get_word_attribution, get_word_attribution_for_training, loss_fn
import json
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
    :param bullshit_words: List of words to be masked out during training. Leaves None for no masking.

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

def fine_tune_with_explanations(train_df, 
                                 base_model="google-bert/bert-base-cased", 
                                 fine_tuned_model_path="model_weights/fine_tuned_bert_with_ex.pth", 
                                 epochs=3, 
                                 batch_size=16, 
                                 learning_rate=2e-5,
                                 bullshit_words=[],
                                 explanation_loss_ratio=0.2,
                                 lam=1.0,
                                 n_steps=500,
                                 checkpoint_every_n_step = 1
                                ):
    """
    Fine tunes the model, using the regular loss alongside the loss of the explanation method.
    
    :param train_df: DataFrame containing training data.
    :param base_model: Path to the HuggingFace mode which is used. Defaults to "google-bert/bert-base-cased".
    :param fine_tuned_model_path: Path to save the fine-tuned model.
    :param epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :param learning_rate: Learning rate for the optimizer.
    :param bullshit_words: List of words to be masked out during explanation-based training.
    :param explanation_loss_ratio: Percentage of the dataset to be trained with explanation loss.
    :param lam: Lambda value to scale the explanation loss.
    :param n_steps: Number of steps for Integrated Gradients.
    :param checkpoint_every_n_step: Creates a checkpoint every n'th batch computed in the second learning phase.

    :return: Path to the fine-tuned model.
    """
    
    checkpoint_path = "model_weights/checkpoint.pth"
    
    tokenizer, model, device = load_base_model(base_model)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Converts the string tokens to BERT compatible input ids 
    bullshit_token_ids = set()
    for word in bullshit_words:
        # Encode without special tokens [CLS]/[SEP] to get the raw IDs
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
        first_phase_avg_loss, second_phase_avg_loss = train_with_explanation_one_step(
            model=model,
            tokenizer=tokenizer,
            train_df=train_df,
            optimizer=optimizer,
            device=device,
            bullshit_ids=bullshit_ids_tensor,
            explanation_loss_ratio=explanation_loss_ratio,
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

def run_attributions(n_steps, 
                     save_every, 
                     internal_batch_size, 
                     tokenizer, 
                     model, 
                     train_df, 
                     output_json="attributions_and_logits/global_word_attributions.json", 
                     review_column="review"):
    """
    Calculates word attributions for all reviews in the provided dataframe using the fine-tuned model.
    Uses Integrated Gradients method from Captum.
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

    :return: output_json: Path to the output JSON file.
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

def train_with_explanation_one_step(
        model : torch.nn.Module,
        tokenizer,
        train_df, 
        optimizer, 
        device,
        bullshit_ids : torch.Tensor,
        explanation_loss_ratio = 0.2, 
        lam = 1.0,
        n_steps = 500,
        batch_size=16,
        checkpoint_every_n_step=1,
        checkpoint_path="model_weights/checkpoint.pth",
        epoch=1
    ):
    """
    Trains the model in two phaes. One regular training and one where we incorporate the additional loss via usage of the bullshit words.
    Comes with checkpointing to be able to recover from crashes. Comes with tqdm progress bars.

    :param model: The model to train.
    :param tokenizer: The tokenizer corresponding to the model.
    :param train_df: DataFrame containing training data.
    :param optimizer: The optimizer to use for training.
    :param device: Either 'cpu' or 'cuda'.
    :param bullshit_ids: list of token IDs that are considered bullshit words and will be punished for using
    :param explanation_loss_ratio: determines the ratio how much of the training set 
            is trained regularly (1.0 - ex_loss_training_data_loader) how much with the extra loss.
    :param n_steps: Number of steps for the Integrated Gradients.
    :param batch_size: Batch size for training.
    :param checkpoint_every_n_step: Save a checkpoint every n steps.
    :param checkpoint_path: Path to save the checkpoint.
    :param epoch: Current epoch number. For checkpointing purposes.
    """
    
    for param in model.bert.embeddings.word_embeddings.parameters():
        param.requires_grad = True
    
    prev_checkpoint = {}
    if os.path.exists(checkpoint_path):
        prev_checkpoint = torch.load(checkpoint_path)
    
    # Index of the last_batch that we trained on before the program crashed
    last_batch_idx : int = 0
    last_batch_idx = prev_checkpoint.get("batch_idx", last_batch_idx)
    
    # Split the training set into two subsets
    split_idx = int(len(train_df) * (1.0 - explanation_loss_ratio)) 
    regular_training_set = train_df[:split_idx]
    regular_training_data_loader = create_data_loader(regular_training_set, tokenizer, batch_size)
    ex_loss_training_set = train_df[split_idx:]
    ex_loss_training_data_loader = create_data_loader(ex_loss_training_set, tokenizer, batch_size)
    
    # First train regularly with the first set
    # We can skip the first phase if the last_batch_idx != 0 because it means that we already made a checkpoint during phase 2
    
    first_phase_avg_loss = 0.0
    if last_batch_idx == 0:
        print("Regular training started")
        first_phase_avg_loss = train_one_step(model, regular_training_data_loader, optimizer, device)
        print("Regular training finished. Starting explanation loss training")
    else:
        print("can skip first learning phase, since last time we already reached the second")
    # Then train with the second set, with the explanation loss
    model.train()
    
    second_phase_total_loss = 0.0
    
    second_phase_total_loss = prev_checkpoint.get("second_phase_total_loss", second_phase_total_loss)
    first_phase_avg_loss = prev_checkpoint.get("first_phase_avg_loss", first_phase_avg_loss)
    
    
    for batch_idx, batch in enumerate(tqdm(ex_loss_training_data_loader)):
        # We skip all the batches from the last (failed) run
        if batch_idx < last_batch_idx:
            continue
        
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        label = batch[2].to(device)
        
        optimizer.zero_grad()
        
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        
        reason, _  = get_word_attribution_for_training(
                                                model=model, 
                                                batch=batch, 
                                                device=device, 
                                                n_steps=n_steps,
                                            )

        loss = loss_fn(output, label, reason, input_ids, bullshit_ids, lam=lam)
        loss.backward()
        optimizer.step()
        
        second_phase_total_loss += loss.item()
        
        if (batch_idx + 1) % checkpoint_every_n_step == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'batch_idx':batch_idx,
                'loss': loss,
                'first_phase_avg_loss': first_phase_avg_loss,
                "second_phase_total_loss": second_phase_total_loss
            }
            temp_filename = checkpoint_path + ".tmp"
            torch.save(checkpoint, temp_filename)
            os.replace(temp_filename, checkpoint_path)
    second_phase_avg_loss = second_phase_total_loss / len(ex_loss_training_data_loader)
    
    # Checkpoint at the end of epoch
    checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
    temp_filename = checkpoint_path + ".tmp"
    torch.save(checkpoint, temp_filename)
    os.replace(temp_filename, checkpoint_path)
    
    return first_phase_avg_loss, second_phase_avg_loss


def extract_logits(train_df, output_json="review_logits.json", review_column="review", batch_size=32):
    """
    Extracts logits from the fine-tuned model for all reviews in the provided dataframe.
    Saves the logits to a JSON file.
    Used for analysis of the integrated gradients results. 

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