import torch
from torch.optim import AdamW
from captum.attr import LayerIntegratedGradients, visualization, IntegratedGradients
from utils import create_data_loader, load_base_model, create_train_test_split, train_one_step
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json
from collections import defaultdict
import numpy as np
import os

def get_word_attribution(n_steps, review: str | list, model, tokenizer, target = 1, internal_batch_size=16):
    """
    Returns a dictionary where the keys are each word in the given sentence and the value is the associated attribution score.

    :param n_steps: Number of steps for the Integrated Gradients.
    :param review: The input review text.
    :param model: The fine-tuned model.
    :param tokenizer: The tokenizer corresponding to the model.
    :param target: The target class for which attributions are computed.
    :param internal_batch_size: Batch size for internal model processing.

    :return: A dictionary mapping words to their attribution scores.
    :returns: delta: convergence delta value from Integrated Gradients.
    """

    def predict(inputs, token_type_ids=None, attention_mask=None):
        return model(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
    
    lig = LayerIntegratedGradients(predict, model.bert.embeddings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenize with max_length=512 to avoid exceeding BERT's max sequence length
    if type(review) == str:
        encoded = tokenizer(review, truncation=True, padding=True, max_length=512, return_tensors='pt')
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
    else:
        input_ids = review[0].to(device)
        attention_mask = review[1].to(device)
        
    baseline_ids = torch.zeros_like(input_ids) # Often 0 is the [PAD] token ID
    # baseline_attention_mask = torch.zeros_like(attention_mask)
    
    attributions, delta = lig.attribute(
                                    inputs=input_ids,
                                    baselines=baseline_ids,
                                    target=target,
                                    return_convergence_delta=True,
                                    n_steps=n_steps,
                                    internal_batch_size=internal_batch_size,
                                    additional_forward_args=(None, attention_mask))

    # Sum across the embedding dimension (dim=2)
    attributions_sum = attributions.sum(dim=-1).squeeze(0)
    # Normalize for visualization
    #attributions_sum = attributions_sum / torch.norm(attributions_sum)

    tokens = []
    for review in input_ids[:]:
        
        tokens.append(tokenizer.convert_ids_to_tokens(review)) 
    
    # result = {}
    
    # for tok, val in zip(tokens, attributions_sum):
    #     result[tok] = val.item()

    return {"tokens":tokens, "attributions":attributions_sum}, delta

def get_word_attribution_for_training(model, batch, device, n_steps=20):
    model.train() # Must be in train mode
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2].to(device)

    # 1. Target the word embeddings sub-module
    target_layer = model.bert.embeddings.word_embeddings

    # 2. Define the wrapper
    def forward_func(inputs, mask):
        # We must extract logits for Captum
        return model(input_ids=inputs, attention_mask=mask).logits

    lig = LayerIntegratedGradients(forward_func, target_layer)

    # 3. Force the model to keep the graph alive
    # We use return_convergence_delta=True to avoid the unpacking error
    attributions, delta = lig.attribute(
        inputs=input_ids,
        target=labels,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        attribute_to_layer_input=False,
        return_convergence_delta=True 
    )

    # Ensure the attribution itself requires grad
    # (Captum usually handles this, but we can sum to verify)
    word_scores = attributions.sum(dim=-1)
    
    return word_scores, delta

# def loss_fn(output : torch.Tensor, labels, word_scores : dict[str, any], bullshit_words : list[str], lam : float = 1.0) -> torch.Tensor:
#     logits = output.logits
    
#     criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
#     individual_loss = criterion(logits, labels)
    
#     attributions = word_scores.get("attributions")
#     tokens = word_scores.get("tokens")
    
#     for review in range(len(tokens)):
#         ex_loss = 0.0
#         for i, t in enumerate(tokens[review]):
#             if not t in bullshit_words:
#                 attributions[review, i] = 0
#             else:
#                 ex_loss += abs(attributions[review, i])
        
#         attributions_sum = ex_loss#attributions[review].sum()
#         individual_loss[review] += lam * attributions_sum
    
#     final_loss = individual_loss.pow(2).mean()

#     return final_loss

def loss_fn(output, labels, attributions, input_ids, bullshit_ids, lam=1.0):
    """
    Args:
        output: The model output object (containing logits)
        labels: Ground truth labels
        attributions: [batch_size, seq_len] tensor from get_word_attribution
        input_ids: [batch_size, seq_len] the original token IDs
        bullshit_ids: [num_bullshit_tokens] tensor of IDs to punish
        lam: The penalty strength
    """
    # 1. Standard CrossEntropy (Classification)
    # Use reduction='mean' (default) for stability
    criterion = torch.nn.CrossEntropyLoss()
    ce_loss = criterion(output.logits, labels)
    
    # 2. Explanation Loss (The Punishment)
    # Create a mask: 1.0 if the token is a 'bullshit' token, 0.0 otherwise
    # torch.isin is very efficient on GPU
    is_bullshit_mask = torch.isin(input_ids, bullshit_ids).float()
    
    # Calculate the sum of absolute attributions for only the bullshit tokens
    # We use .abs() because both positive and negative attribution 
    # indicate the model is 'using' that word to make its decision.
    attribution_penalty = (attributions.abs() * is_bullshit_mask).sum(dim=-1).mean()
    # if attribution_penalty.item() != 0.0:
        # breakpoint
    
    # Total loss
    return ce_loss + (lam * attribution_penalty)

    
def training_with_explanaition_batched_test_run():
    "method to test, training the model on a single example review, where we also use the IG to adjust the training of the model"
    
    #load the model
    tokenizer, model, _ = load_base_model()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("fine_tuned_bert.pth", map_location=device))
    model.to(device)

    bullshit_words = ["superb", "fantastic", "best"]
    
    
    reviews, _ = create_train_test_split()
    data_loader = create_data_loader(reviews, tokenizer, 8)
    #load the reviews for training
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()
    
    for batch in tqdm(data_loader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        label = batch[2].to(device)
        
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        reason, delta  = get_word_attribution(500, batch, model, tokenizer)
        
        loss = loss_fn(output, label, reason, bullshit_words)
        loss.backward()
        optimizer.step()
        #based on that reason adjust the loss function

def train_with_explaination_one_step(
        model : torch.nn.Module,
        tokenizer,
        train_df, 
        optimizer, 
        device,
        bullshit_ids : torch.Tensor,
        explanaition_loss_ratio = 0.2, 
        lam = 1.0,
        n_steps = 500,
        batch_size=16,
        checkpoint_every_n_step=1,
        checkpoint_path="checkpoint.pth",
        epoch=1
    ):
    """trains the model in two phaes. One regular training and one where we incorporate the additional loss via usage of the bullshit words

    Args:
        explanaition_loss_ratio (float, optional): determines the raio how much of the training set 
            is trained regularly (1.0 - ex_loss_training_data_loader) how much with the extra loss. Defaults to 0.2.
        lam (float, optional): lambda for the explanaition loss
        bullshit_ids (list): list of strings that are considered bullshit words and will be punished for using
    """
    
    for param in model.bert.embeddings.word_embeddings.parameters():
        param.requires_grad = True
    
    prev_checkpoint = {}
    if os.path.exists(checkpoint_path):
        prev_checkpoint = torch.load(checkpoint_path)
    
    #index of the last_batch that we trained on before the program crashed
    last_batch_idx : int = 0
    last_batch_idx = prev_checkpoint.get("batch_idx", last_batch_idx)
    
    #split the training set into two subsets
    split_idx = int(len(train_df) * (1.0 - explanaition_loss_ratio)) 
    regular_training_set = train_df[:split_idx]
    regular_training_data_loader = create_data_loader(regular_training_set, tokenizer, batch_size)
    ex_loss_training_set = train_df[split_idx:]
    ex_loss_training_data_loader = create_data_loader(ex_loss_training_set, tokenizer, batch_size)
    
    #first train regularly with the first set
    ##we can skip the first phase if the last_batch_idx != 0 because it means that we already made a checkpoint during phase 2
    
    first_phase_avg_loss = 0.0
    if last_batch_idx == 0:
        print("Regular training started")
        first_phase_avg_loss = train_one_step(model, regular_training_data_loader, optimizer, device)
        print("Regular training finished. Starting explanaition loss training")
    else:
        print("can skip first learning phase, since last time we already reached the second")
    #then tain with the second set, with the explanaition loss
    model.train()
    
    second_phase_total_loss = 0.0
    
    second_phase_total_loss = prev_checkpoint.get("second_phase_total_loss", second_phase_total_loss)
    first_phase_avg_loss = prev_checkpoint.get("first_phase_avg_loss", first_phase_avg_loss)
    
    
    for batch_idx, batch in enumerate(tqdm(ex_loss_training_data_loader)):
        #we skip all the batches from the last (failed) run
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
        # reason.requires_grad = True

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
                # Add anything else you need (e.g., scheduler state)
            }
            temp_filename = checkpoint_path + ".tmp"
            torch.save(checkpoint, temp_filename)
            os.replace(temp_filename, checkpoint_path)
    second_phase_avg_loss = second_phase_total_loss / len(ex_loss_training_data_loader)
    
    ##at the end of the epoch we make a final checkpoint where we set to the next epoch
    checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                # Add anything else you need (e.g., scheduler state)
            }
    temp_filename = checkpoint_path + ".tmp"
    torch.save(checkpoint, temp_filename)
    os.replace(temp_filename, checkpoint_path)
    
    return first_phase_avg_loss, second_phase_avg_loss

def training_with_explanaition_test_run():
    "method to test, training the model on a single example review, where we also use the IG to adjust the training of the model"
    
    #load the model
    tokenizer, model, _ = load_base_model()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("fine_tuned_bert.pth", map_location=device))
    model.to(device)

    bullshit_words = ["superb", "fantastic", "best"]
    
    #load the reviews for training
    reviews = [
        "This movie was absolutely fantastic and the acting was superb.",
        "This was the best movie in the world"
    ]
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()
    
    def print_sorted_token_ranking(feature_attribution : dict):
            asc = {k: v for k, v in reversed(sorted(feature_attribution.items(), key=lambda item: item[1]))}
            for k, v in asc.items():
                print(f"token: {k} = {v}")    
    
    for i, review in enumerate(reviews):
        print(f"\nreview {i}:\n")
        
        optimizer.zero_grad()
        
        #get the loss and outcome of the model
        inputs = tokenizer(review, return_tensors='pt', truncation=True, padding=True).to(device)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        label = torch.tensor([0]).to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        
        logits = output.logits
        prediction_id = torch.argmax(logits, dim=-1).item()
        label = model.config.id2label[prediction_id]
        
        #compute word_score
        word_scores = get_word_attribution(5, review, model, tokenizer)
        word_scores.requires_grad()
        
        #compute final loss with bullshit words
        loss = output.loss
        loss = loss_fn(output, word_scores, bullshit_words)
        loss.backward()
        optimizer.step()
        
        
        print_sorted_token_ranking(word_scores)

def model_evaluation(model, test_df, tokenizer, device):

    """
    Evaluates a model on the provided test dataframe and returns classification report and confusion matrix.

    :param model: The model to evaluate.
    :param test_df: DataFrame containing testing data.
    :param tokenizer: The tokenizer corresponding to the model.
    :param device: Either 'cpu' or 'cuda'.

    :return: classification_report, confusion_matrix
    """

    model.eval()

    test_loader = create_data_loader(test_df, tokenizer, batch_size=16)

    all_predictions = []
    all_true_labels = []    

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch] 
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.argmax(outputs.logits, dim=-1) # Get predicted class

            # Move predictions and true labels to cpu for metric calculation
            all_predictions.extend(preds.cpu().numpy()) 
            all_true_labels.extend(labels.cpu().numpy()) 

    # Calculate classification metrics
    class_report = classification_report(all_true_labels, all_predictions, target_names=['negative', 'positive'], output_dict=True)
    confus_mat = confusion_matrix(all_true_labels, all_predictions)

    return class_report, confus_mat

def get_most_meaningful_words(attribution_values_json, top_n=10, absolute=True, threshold=50):
    """
    Extracts the top_n most meaningful words based on their average attribution scores.

    :param attribution_values_json: JSON file containing word attributions.
    :param top_n: Number of top words to extract.
    :param absolute: Whether to consider absolute attribution scores for sorting.
    :param thereshold: Minimum number of occurrences for a word to be considered.

    :return: List of top_n words with highest attributions and None. If absolute is False, 
             returns two lists for positive and negative attributions.
    """

    # Metadata
    stats_json = attribution_values_json.replace(".json", "_stats.json")
    with open(stats_json, 'r') as f:
        stats = json.load(f)

    # Attributions
    with open(attribution_values_json, 'r') as f:
        data = json.load(f)

    total_occurrences = stats.get("word_counts", {})

    # Only consider words that appear at least 'threshold' times
    def threshold_passing(word_item):
        word = word_item[0]
        count = total_occurrences.get(word, 0)
        return count >= threshold
        
    # Filter words based on occurrence threshold
    filtered_items = filter(threshold_passing, data.items())

    if absolute:
    
        # Sort by absolute attribution values
        sorted_keys = sorted(filtered_items, key=lambda item: abs(item[1]), reverse=True)

        return sorted_keys[:top_n], None
    else:

        # Sort separately for positive and negative attributions
        sorted_keys_positive = sorted(filtered_items, key=lambda item: item[1], reverse=True)
        sorted_keys_positive = [item for item in sorted_keys_positive if item[1] > 0]

        sorted_keys_negative = sorted(filtered_items, key=lambda item: item[1])
        sorted_keys_negative = [item for item in sorted_keys_negative if item[1] < 0]

        return sorted_keys_positive[:top_n], sorted_keys_negative[:top_n]
    

def calculate_relative_error(logits_json_path="review_logits.json", stats_json_path="global_word_attributions_stats.json"):
    """
    Calculates the prozentual error between the IG-procedure in relation to hte models output logits.

    :param logits_json_path: Path to the JSON file containing review logits.
    :param stats_json_path: Path to the JSON file containing global word attribution statistics.

    :return: relative_error: float
    """

    with open(stats_json_path, 'r') as f:
        stats = json.load(f)

    with open(logits_json_path, 'r') as f:
        logits_data = json.load(f)

    # Delta values from stats
    total_abs_delta = stats["total_abs_delta"]
    reviews_processed = stats["reviews_processed"]
    raw_avg_delta = total_abs_delta / reviews_processed

    # Extract predicted class logits
    pred_classes = []
    for entry in logits_data:
        
        pred_class = entry["prediction"]
        pred_class_logit = entry[f"logit_{pred_class}"]

        pred_classes.append(pred_class_logit)

    # Get average logit magnitude
    avg_logit_magnitude = np.mean(pred_classes) 
    relative_error = raw_avg_delta / avg_logit_magnitude

    return relative_error


if __name__ == '__main__':
    # training_with_explanaition_test_run()
    training_with_explanaition_batched_test_run()
    # relative_error = calculate_relative_error()
    # print(relative_error)