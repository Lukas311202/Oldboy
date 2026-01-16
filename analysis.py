import torch
from torch.optim import AdamW
from captum.attr import LayerIntegratedGradients, visualization
from utils import create_data_loader, load_base_model
from sklearn.metrics import classification_report, confusion_matrix

# def predict(inputs, token_type_ids=None, attention_mask=None):
    # Returns the logit/score for the target class (e.g., index 1 for positive)
    # return model(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

# Wrap the model's embedding layer
# tokenizer, model, _ = load_model("google-bert/bert-base-cased")
# lig = LayerIntegratedGradients(predict, model.bert.embeddings)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.load_state_dict(torch.load("fine_tuned_bert.pth", map_location=device))
# model.to(device)
# model.eval() # Set to evaluation mode!

# review = "This movie was absolutely fantastic and the acting was superb."

# input_ids = tokenizer.encode(review, return_tensors='pt').to(device)
# baseline_ids = torch.zeros_like(input_ids) # Often 0 is the [PAD] token ID

# attributions, delta = lig.attribute(inputs=input_ids,
#                                     baselines=baseline_ids,
#                                     target=1, # Index of the 'Positive' class
#                                     return_convergence_delta=True)

# # Sum across the embedding dimension (dim=2)
# attributions_sum = attributions.sum(dim=-1).squeeze(0)
# # Normalize for visualization
# attributions_sum = attributions_sum / torch.norm(attributions_sum)

# tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# for tok, val in zip(tokens, attributions_sum):
#     print(f"{tok}: {val}")


def get_word_attribution(n_steps, review: str, model, tokenizer, target = 1):
    """
    Returns a dictionary where the keys are each word in the given sentence and the value is the associated attribution score.

    :param n_steps: Number of steps for the Integrated Gradients.
    :param review: The input review text.
    :param model: The fine-tuned model.
    :param tokenizer: The tokenizer corresponding to the model.

    :return: A dictionary mapping words to their attribution scores.
    :returns: delta: convergence delta value from Integrated Gradients.
    """

    def predict(inputs, token_type_ids=None, attention_mask=None):
        return model(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
    
    lig = LayerIntegratedGradients(predict, model.bert.embeddings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenize with max_length=512 to avoid exceeding BERT's max sequence length
    encoded = tokenizer(review, truncation=True, padding=True, max_length=512, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    baseline_ids = torch.zeros_like(input_ids) # Often 0 is the [PAD] token ID
    # baseline_attention_mask = torch.zeros_like(attention_mask)
    
    attributions, delta = lig.attribute(inputs=input_ids,
                                    baselines=baseline_ids,
                                    target=target,
                                    return_convergence_delta=True,
                                    n_steps=n_steps,
                                    additional_forward_args=(None, attention_mask))

    # Sum across the embedding dimension (dim=2)
    attributions_sum = attributions.sum(dim=-1).squeeze(0)
    # Normalize for visualization
    #attributions_sum = attributions_sum / torch.norm(attributions_sum)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    result = {}
    
    for tok, val in zip(tokens, attributions_sum):
        result[tok] = val.item()

    return result, delta.item()

def loss_fn(output, word_scores : dict[str, float], bullshit_words : list[str]):
    res = output.loss
    
    found_bs :int = 0
    for bs in bullshit_words:
        bs_loss = word_scores.get(bs, 0.0)
        if bs_loss:
            found_bs += 1
        res += bs_loss
    print(f"found {found_bs} bullshit words")
    return res

    """method to test, training the model on a single example review, where we also use the IG to adjust the training of the model 
    """
def training_with_explanaition_test_run():
    
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
        word_scores = get_word_attribution(review, model, tokenizer)
        
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


if __name__ == '__main__':
    training_with_explanaition_test_run()