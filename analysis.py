import torch
from captum.attr import LayerIntegratedGradients, visualization
from utils import load_model

def predict(inputs, token_type_ids=None, attention_mask=None):
    # Returns the logit/score for the target class (e.g., index 1 for positive)
    return model(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

# Wrap the model's embedding layer
tokenizer, model, _ = load_model("google-bert/bert-base-cased")
lig = LayerIntegratedGradients(predict, model.bert.embeddings)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("fine_tuned_bert.pth", map_location=device))
model.to(device)
model.eval() # Set to evaluation mode!

review = "This movie was absolutely fantastic and the acting was superb."

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

print(get_word_attribution(review, model, tokenizer))
print(get_word_attribution("This movie was shit", model, tokenizer))