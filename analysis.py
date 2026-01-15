import torch
from torch.optim import AdamW
from captum.attr import LayerIntegratedGradients, visualization
from utils import load_model

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


"""returns a dictionary where the keys is each word in the given sentence and the value is the associated 
"""
def get_word_attribution(review: str, model, tokenizer, target = 1):
    
    def predict(inputs, token_type_ids=None, attention_mask=None):
        return model(inputs, token_type_ids, attention_mask)[0]
    
    lig = LayerIntegratedGradients(predict, model.bert.embeddings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = tokenizer.encode(review, return_tensors='pt').to(device)
    baseline_ids = torch.zeros_like(input_ids) # Often 0 is the [PAD] token ID
    
    attributions, delta = lig.attribute(inputs=input_ids,
                                    baselines=baseline_ids,
                                    target=target,
                                    return_convergence_delta=True)

    # Sum across the embedding dimension (dim=2)
    attributions_sum = attributions.sum(dim=-1).squeeze(0)
    # Normalize for visualization
    attributions_sum = attributions_sum / torch.norm(attributions_sum)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    result = {}
    
    for tok, val in zip(tokens, attributions_sum):
        result[tok] = val
    return result

def loss_fn(output, word_scores : dict[str, float], bullshit_words : list[str]):
    res = output.loss
    
    found_bs :int = 0
    for bs in bullshit_words:
        bs_loss = word_scores.get(bs, 0.0)
        if bs_loss:
            found_bs += 1
        res -= bs_loss
    print(f"found {found_bs} bullshit words")
    return res

    """method to test, training the model on a single example review, where we also use the IG to adjust the training of the model 
    """
def training_with_explanaition_test_run():
    
    #load the model
    tokenizer, model, _ = load_model()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("fine_tuned_bert.pth", map_location=device))
    model.to(device)
    model.train()

    bullshit_words = ["superb", "fantastic", "best"]
    
    #load the reviews for training
    # review = "This movie was absolutely fantastic and the acting was superb."
    reviews = [
        "This movie was absolutely fantastic and the acting was superb.",
        "This was the best movie in the world"
    ]
    # inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True).to(device)
    
    # output = model(**inputs)
    # logits = output.logits
    
    # prediction_id = torch.argmax(logits, dim=-1).item()
    # label = model.config.id2label[prediction_id]
    # prediction_id = torch.argmax(logits).item()
    # label = model.config.id2label[prediction_id]
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    def print_sorted_token_ranking(feature_attribution : dict):
            asc = {k: v for k, v in reversed(sorted(feature_attribution.items(), key=lambda item: item[1]))}
            for k, v in asc.items():
                print(f"token: {k} = {v}")    
    
    for i, review in enumerate(reviews):
        print(f"\nreview {i}:\n")
        
        optimizer.zero_grad()
        
        #get the loss and outcome of the model
        inputs = tokenizer(review, return_tensors='pt', truncation=True, padding=True).to(device)
        label = torch.tensor([1]).to(device)
        output = model(inputs.input_ids, inputs.attention_mask, label)
        
        #compute word_score
        word_scores = get_word_attribution(review, model, tokenizer)
        
        #compute final loss with bullshit words
        loss = loss_fn(output, word_scores, bullshit_words)
        loss.backward()
        optimizer.step()
        
        
        print_sorted_token_ranking(word_scores)

if __name__ == '__main__':
    training_with_explanaition_test_run()