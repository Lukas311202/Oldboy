import torch
from utils import load_model
"""
ONLY FOR TESTING PURPOSES
WILL BE DELETED LATER!!!!!!
"""

# 1. Setup
tokenizer, model, device = load_model("google-bert/bert-base-cased")

# 2. Test review
text = "Worst movie ever. Completely boring and a waste of time."

# 3. Tokenization
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

# 4. Prediction (without gradient)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    
print(logits)

# 5. Interpret result
# We take the highest value (Argmax)
prediction_id = torch.argmax(logits, dim=-1).item()
label = model.config.id2label[prediction_id]

print(f"Text: {text}")
print(f"Predicted Sentiment: {label}")