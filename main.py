from analysis import model_evaluation, get_most_meaningful_words
from model_loops import fine_tune_loop, run_attributions, fine_tune_with_explanaitions
from torch.optim import AdamW 
import torch
from utils import create_train_test_split, load_fine_tuned_model, load_fine_tuned_model
from plotting import (
    plot_confusion_matrix,
    plot_classification_report,
    plot_overall_metrics,
    plot_overall_metrics_comparison,
    plot_classification_report_comparison,
    plot_confusion_matrix_difference
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
    fine_tuned_model_path = "model_weights/fine_tuned_bert.pth"
    # Load the fine-tuned model 
    tokenizer, fine_tuned_model, device = load_fine_tuned_model(model_name=MODEL_NAME, model_path=fine_tuned_model_path)
    # Evaluate
    classification_report, confusion_matrix = model_evaluation(model=fine_tuned_model, test_df=test_df, tokenizer=tokenizer, device=device)

    CLASS_NAMES = ["negative", "positive"]
    subdir = "baseline"
    # Confusion matrix
    plot_confusion_matrix(
        confusion_matrix, 
        CLASS_NAMES,
        subdir=subdir
    )

    plot_confusion_matrix(
        confusion_matrix, 
        CLASS_NAMES, 
        normalize=True,
        subdir=subdir
    )

    # Classification report metrics
    plot_classification_report(
        classification_report, 
        CLASS_NAMES,
        subdir=subdir
    )

    # Overall performance metrics
    plot_overall_metrics(classification_report, subdir=subdir)

    # Reduce dataset for attribution calculations
    reduced_df, _ = train_test_split(
        train_df, 
        train_size=10000, 
        stratify=train_df['sentiment'], 
        random_state=42
    )

    # Calculate word attributions
    #attribution_values_json, final_avg_delta = run_attributions(n_steps=30, save_every=15, internal_batch_size=80, tokenizer=tokenizer, model=fine_tuned_model, train_df=reduced_df) 
    # NOTE: IN REPORT SCHREIBEN WARUM WIR N_STEPS GENOMMEN HABEN (Original paper zitieren) Mit delta value (delta sollte < 0.05 sein laut paper um gute attributionen zu haben. Das sind 20 bis 300 steps)
    attribution_values_json = "global_word_attributions.json"

    # Get most meaningful words
    most_meaningful_words, _ = get_most_meaningful_words(attribution_values_json, top_n=200, absolute=True, threshold=100)
    for word, score in most_meaningful_words:
        print(f"Word: {word}, Score: {score}")

    # Decide which words to consider as bullshit words
    bullshit_words = [
        "Zero", "Rather", "turkey", "sheer", "hi", "episodes", "episode", "today", "flu",
        "summary", "sports", "Kid", "block", "Sinatra", "born", "Anyone", "entry", "Although", "Bourne",
        "app", "DVD", "short", "animated", "Yet", "Many", "Not", "scenery", "beginning", "Day", "bit",
        "adult", "describe", "true", "personally", "ready", "match"
    ]

    # Fine-tune again with explanation-based loss
    """ex_model_path = fine_tune_with_explanaitions(train_df, 
                                 n_steps=500, 
                                 batch_size=80, 
                                 epochs=3,
                                 bullshit_words=bullshit_words,
                                 checkpoint_every_n_step=5,
                                 lam=1.0,
                                 fine_tuned_model_path="model_weights/fine_tuned_bert_with_ex.pth"
                                )
    """

    ex_model_path = "model_weights/fine_tuned_bert_with_ex.pth"

    # Load the fine-tuned model with explanations
    _, ex_model, _ = load_fine_tuned_model(
        model_name=MODEL_NAME,
        model_path=ex_model_path
    )

    # Evaluate the fine tuned model with explanations
    ex_classification_report, ex_confusion_matrix = model_evaluation(
        model=ex_model,
        test_df=test_df,
        tokenizer=tokenizer,
        device=device
    )

    subdir = "with_explanations"

    # Confusion matrix
    plot_confusion_matrix(
        ex_confusion_matrix, 
        CLASS_NAMES,
        subdir=subdir
    )

    plot_confusion_matrix(
        ex_confusion_matrix, 
        CLASS_NAMES, 
        subdir=subdir,
        normalize=True
    )

    # Classification report metrics
    plot_classification_report(
        ex_classification_report, 
        CLASS_NAMES,
        subdir=subdir
    )

    # Overall performance metrics
    plot_overall_metrics(ex_classification_report, subdir=subdir)

    subdir = "comparisons"

    plot_overall_metrics_comparison(
        classification_report,
        ex_classification_report,
        label_a="Baseline",
        label_b="With Explanation",
        subdir=subdir
    )

    plot_classification_report_comparison(
        classification_report,
        ex_classification_report,
        CLASS_NAMES,
        label_a="Baseline",
        label_b="With Explanation",
        subdir=subdir
    )

    plot_confusion_matrix_difference(
        confusion_matrix,
        ex_confusion_matrix,
        CLASS_NAMES,
        subdir=subdir
    )

    # Load model with masking out bullshit words fine-tuning
    #mask_path = fine_tune_loop(train_df=train_df, bullshit_words=bullshit_words, fine_tuned_model_path="model_weights/test_fine_tuned_bert_masking.pth")
    mask_path = "model_weights/test_fine_tuned_bert_masking.pth"

    tokenizer, model, device = load_fine_tuned_model(model_path=mask_path)

    class_report_masked, confus_matrix_masked = model_evaluation(model=model, test_df=test_df, tokenizer=tokenizer, device=device)

    subdir = "masking"

    # Confusion matrix
    plot_confusion_matrix(
        confus_matrix_masked, 
        CLASS_NAMES,
        subdir=subdir
    )

    plot_confusion_matrix(
        confus_matrix_masked, 
        CLASS_NAMES, 
        subdir=subdir,
        normalize=True
    )

    # Classification report metrics
    plot_classification_report(
        class_report_masked, 
        CLASS_NAMES,
        subdir=subdir
    )

    # Overall performance metrics
    plot_overall_metrics(class_report_masked, subdir=subdir)
    

