from analysis import model_evaluation, get_most_meaningful_words
from model_loops import fine_tune_loop, run_attributions
from torch.optim import AdamW 
import torch
from utils import create_train_test_split, load_fine_tuned_model, load_fine_tuned_model
from plotting import (
    plot_confusion_matrix,
    plot_classification_report,
    plot_overall_metrics
)
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    print("torch version:", torch.__version__)
    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    else:
        print("CUDA not found. Check your drivers.")

    MODEL_NAME = "google-bert/bert-base-cased"
    DATA_PATH = "imdb_dataset.csv"
    # Train-test split
    train_df, test_df = create_train_test_split(data=DATA_PATH, text_column="review", label_column="sentiment",
                                                test_size=0.2, seed=42, stratify=True)
    # Fine-tune the model
    #fine_tuned_model_path = fine_tune_loop(train_df=train_df, base_model=MODEL_NAME, epochs=3, batch_size=16, learning_rate=2e-5)
    fine_tuned_model_path = "fine_tuned_bert.pth"
    # Load the fine-tuned model 
    tokenizer, fine_tuned_model, device = load_fine_tuned_model(model_name=MODEL_NAME, model_path=fine_tuned_model_path)
    # Evaluate
    classification_report, confusion_matrix = model_evaluation(model=fine_tuned_model, test_df=test_df, tokenizer=tokenizer, device=device)

    CLASS_NAMES = ["negative", "positive"]

    # Confusion matrix
    plot_confusion_matrix(confusion_matrix, CLASS_NAMES)
    plot_confusion_matrix(confusion_matrix, CLASS_NAMES, normalize=True)
    # Classification report metrics
    plot_classification_report(classification_report, CLASS_NAMES)
    # Overall performance metrics
    plot_overall_metrics(classification_report)

    # Reduce dataset for attribution calculations
    reduced_df, _ = train_test_split(
        train_df, 
        train_size=10000, 
        stratify=train_df['sentiment'], 
        random_state=42
    )

    # Calculate word attributions
    attribution_values_json, final_avg_delta = run_attributions(n_steps=500, save_every=15, internal_batch_size=32, tokenizer=tokenizer, model=fine_tuned_model, train_df=reduced_df) 
    # NOTE: IN REPORT SCHREIBEN WARUM WIR N_STEPS GENOMMEN HABEN (Original paper zitieren) Mit delta value (delta sollte < 0.05 sein laut paper um gute attributionen zu haben. Das sind 20 bis 300 steps)

    # Get most meaningful words
    most_meaningful_words, _ = get_most_meaningful_words(attribution_values_json, top_n=200, absolute=True, threshold=100)

    for word, score in most_meaningful_words:
        print(f"Word: {word}, Score: {score}")

    # TODO: Logit function fertig schreiben um delta besser diskutieren zu können 
    """
    Council decision, which words to consider as bullshit words
    """

    """
    Loss function that incorporates word attributions to penalize the model for relying on bullshit words
    Lambda tweaking
    """



    





