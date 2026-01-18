import torch
from torch.optim import AdamW
from captum.attr import LayerIntegratedGradients, visualization
from utils import create_data_loader, load_base_model, create_train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json
from collections import defaultdict

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
    
    attributions, delta = lig.attribute(inputs=input_ids,
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

def loss_fn(output : torch.Tensor, labels, word_scores : dict[str, any], bullshit_words : list[str]) -> torch.Tensor:
    logits = output.logits
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    individual_loss = criterion(logits, labels)
    
    attributions = word_scores.get("attributions")
    tokens = word_scores.get("tokens")
    
    for review in range(len(tokens)):
        for i, t in enumerate(tokens[review]):
            if not t in bullshit_words:
                attributions[review, i] = 0
    
        attributions_sum = attributions[review].sum()
        individual_loss[review] += attributions_sum
    
    final_loss = individual_loss.pow(2).mean()
    
    # found_bs :int = 0
    # for bs in bullshit_words:
    #     bs_loss = word_scores.get(bs, 0.0)
    #     if bs_loss:
    #         found_bs += 1
    #     res += bs_loss
    # print(f"found {found_bs} bullshit words")
    return final_loss

    
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

if __name__ == '__main__':
    # training_with_explanaition_test_run()
    training_with_explanaition_batched_test_run()