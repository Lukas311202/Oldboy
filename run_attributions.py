import pandas as pd
from collections import defaultdict
import json
from tqdm import tqdm

from main import load_model_and_tokenizer
from analysis import get_word_attribution

tokenizer, model, device = load_model_and_tokenizer()

CSV_PATH = "IMDB_dataset.csv"       
REVIEW_COLUMN = "review"             
OUTPUT_JSON = "global_word_attributions.json"  

print(f"Lade CSV-Datei: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
reviews = df[REVIEW_COLUMN].tolist()
print(f"Anzahl Reviews: {len(reviews)}")

word_sums = defaultdict(float)
word_counts = defaultdict(int)

print("Berechne Word Attributions...")
for review in tqdm(reviews):
    word_attr = get_word_attribution(review, model, tokenizer)
    for word, value in word_attr.items():
        word_sums[word] += value
        word_counts[word] += 1

print("Berechne Durchschnittswerte...")
word_avg = {word: word_sums[word] / word_counts[word] for word in word_sums}

print(f"Speichere Ergebnis in {OUTPUT_JSON}")
with open(OUTPUT_JSON, "w") as f:
    json.dump(word_avg, f, indent=4)

print("Fertig!")
