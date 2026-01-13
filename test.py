import torch
from utils import load_model
"""
ONLY FOR TESTING PURPOSES
WILL BE DELETED LATER!!!!!!
"""

# Setup
tokenizer, model, device = load_model("google-bert/bert-base-cased")

# Test review
text = "Worst movie ever. Completely boring and a waste of time."

# Tokenization
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

# Prediction (without gradient)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    
print(logits)

# Interpret result
# We take the highest value (Argmax)
prediction_id = torch.argmax(logits, dim=-1).item()
label = model.config.id2label[prediction_id]

# Softmax probabilities
probabilities = torch.nn.functional.softmax(logits, dim=-1)


print(f"Text: {text}")
print(f"Probabilities: Negative={probabilities[0][0].item():.4f}, Positive={probabilities[0][1].item():.4f}")
print(f"Predicted Sentiment: {label}")
