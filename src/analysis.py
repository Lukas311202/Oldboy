import torch
from captum.attr import LayerIntegratedGradients
from .utils import create_data_loader
from sklearn.metrics import classification_report, confusion_matrix
import json
import numpy as np

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
    :return: delta: convergence delta value from Integrated Gradients.
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

    tokens = []
    for review in input_ids[:]:
        tokens.append(tokenizer.convert_ids_to_tokens(review)) 

    return {"tokens": tokens, "attributions": attributions_sum}, delta

def get_word_attribution_for_training(model, batch, device, n_steps=20):
    """
    Computes word attributions for a given batch using Layer Integrated Gradients.

    :param model: The fine-tuned model.
    :param batch: 3-Tuple of input_ids, attention_mask, and labels.
    :param device: Either 'cpu' or 'cuda'.
    :param n_steps: Number of steps for the Integrated Gradients.

    :return: word_scores: [batch_size, seq_len] 
    :return: delta: convergence delta value from Integrated Gradients.
    """

    model.train() # Must be in train mode
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2].to(device)

    # Target the word embeddings sub-module
    target_layer = model.bert.embeddings.word_embeddings

    # Extract logits for Captum
    def forward_func(inputs, mask):
        return model(input_ids=inputs, attention_mask=mask).logits

    lig = LayerIntegratedGradients(forward_func, target_layer)

    # Force the model to keep the graph alive
    attributions, delta = lig.attribute(
        inputs=input_ids,
        target=labels,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        attribute_to_layer_input=False,
        return_convergence_delta=True # Avoid the unpacking error
    )

    # Ensure the attribution itself requires grad
    word_scores = attributions.sum(dim=-1)
    
    return word_scores, delta

def loss_fn(output, labels, attributions, input_ids, bullshit_ids, lam=1.0):
    """
    Custom loss function combining CrossEntropy loss with an explanation-based penalty.
    Create a loss that penalizes the model for attributing importance to 'bullshit' words.

    :param output: The model output object (containing logits)
    :param labels: Ground truth labels
    :param attributions: [batch_size, seq_len] tensor from get_word_attribution
    :param input_ids: [batch_size, seq_len] the original token IDs
    :param bullshit_ids: [num_bullshit_tokens] tensor of IDs to punish
    :param lam: The penalty strength

    :return: Combined loss value.
    """

    # Standard CrossEntropy Loss
    criterion = torch.nn.CrossEntropyLoss()
    ce_loss = criterion(output.logits, labels)
    
    # Explanation Loss: Create a mask: 1.0 if the token is a 'bullshit' token, 0.0 otherwise
    is_bullshit_mask = torch.isin(input_ids, bullshit_ids).float()
    
    # Calculate the sum of absolute attributions for only the bullshit tokens
    attribution_penalty = (attributions.abs() * is_bullshit_mask).sum(dim=-1).mean()
    
    # Total loss
    return ce_loss + (lam * attribution_penalty)


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

    test_loader = create_data_loader(test_df, tokenizer, batch_size=16, for_training=False)

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
    

def calculate_relative_error(logits_json_path="output/logs/review_logits.json", stats_json_path="output/logs/global_word_attributions_stats.json"):
    """
    Calculates the relative error between the IG-procedure in relation to the models output logits.

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

def average_jsonl_results(input_file, output_file):
    """
    Averages classification reports and confusion matrices from multiple runs stored in a JSONL file.
    And returns the averaged results as classification report and confusion matrix.

    :param input_file: Path to the input JSONL file containing results from multiple runs.
    :param output_file: Path to the output JSON file to save the averaged results.

    :return: avg_report: Averaged classification report.
    :return: avg_cm: Averaged confusion matrix.
    """
    # Load all results
    results = []
    with open(input_file, "r") as f:
        for line in f:
            results.append(json.loads(line))

    n = len(results)

    avg_report = {}
    categories = ["negative", "positive", "macro avg", "weighted avg"]
    metrics = ["precision", "recall", "f1-score"]
    
    # Initialize average report structure
    for cat in categories:
        avg_report[cat] = {m: 0.0 for m in metrics}
    avg_report["accuracy"] = 0.0
    avg_cm = np.zeros((2, 2))

    # Aggregate results
    for run in results:
        rep = run["classification_report"]
        for cat in categories:
            for m in metrics:
                avg_report[cat][m] += rep[cat][m]
        avg_report["accuracy"] += rep["accuracy"]
        avg_cm += np.array(run["confusion_matrix"])

    # Compute averages
    for cat in categories:
        for m in metrics:
            avg_report[cat][m] /= n
        avg_report[cat]["support"] = results[0]["classification_report"][cat]["support"]
    
    # Finalize accuracy and confusion matrix
    avg_report["accuracy"] /= n
    avg_cm = (avg_cm / n).tolist() # Back to list for JSON serialization

    # Save averaged results
    final_data = {
        "num_runs": n,
        "average_classification_report": avg_report,
        "average_confusion_matrix": avg_cm
    }
    
    # Write to output file
    with open(output_file, "w") as f:
        json.dump(final_data, f, indent=4)
        
    return avg_report, avg_cm
    