from src.analysis import model_evaluation, get_most_meaningful_words
from src.training import fine_tune_loop, run_attributions, fine_tune_with_explanations
import torch
from src.utils import create_train_test_split, load_fine_tuned_model, load_fine_tuned_model, set_seed
from src.plotting import (
    comparison_plots,
    multiple_plots,
)
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    set_seed(42)
    
    print("torch version:", torch.__version__)
    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    else:
        print("CUDA not found. Check your drivers.")
    
    MODEL_NAME = "google-bert/bert-base-cased"
    DATA_PATH = "data/imdb_dataset.csv"

    # Train-test split
    train_df, test_df = create_train_test_split(data=DATA_PATH, label_column="sentiment",
                                                test_size=0.2, seed=42, stratify=True)
    
    # Fine-tune the model
    fine_tuned_model_path = fine_tune_loop(train_df=train_df, base_model=MODEL_NAME, epochs=3, batch_size=16, learning_rate=2e-5)

    # fine_tuned_model_path = "output/model_weights/fine_tuned_bert.pth"
    
    # Load the fine-tuned model 
    tokenizer, fine_tuned_model, device = load_fine_tuned_model(model_name=MODEL_NAME, model_path=fine_tuned_model_path)

    # Evaluate fine-tuned model
    classification_report, confusion_matrix = model_evaluation(model=fine_tuned_model, test_df=test_df, tokenizer=tokenizer, device=device)

    # Plot the baseline results and save them in "output/plots/baseline/" directory
    multiple_plots(subdir="baseline", cm=confusion_matrix, classification_report=classification_report)

    # Reduce dataset for attribution calculations
    reduced_df, _ = train_test_split(
        train_df, 
        train_size=10000, 
        stratify=train_df['sentiment'], 
        random_state=42
    )

    # Calculate word attributions
    attribution_values_json, final_avg_delta = run_attributions(n_steps=30, save_every=15, internal_batch_size=80, tokenizer=tokenizer, model=fine_tuned_model, train_df=reduced_df) 

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
    ex_model_path = fine_tune_with_explanations(train_df, 
                                 n_steps=500, 
                                 batch_size=80, 
                                 epochs=3,
                                 bullshit_words=bullshit_words,
                                 checkpoint_every_n_step=5,
                                 lam=1.0,
                                 fine_tuned_model_path="output/model_weights/fine_tuned_bert_with_ex.pth"
                                )

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

    # Plot the results with explanations and save them in "output/plots/with_explanations/" directory
    multiple_plots(subdir="with_explanations", cm=ex_confusion_matrix, classification_report=ex_classification_report)
 
    # Comparison plots between baseline and explanation-based model
    comparison_plots(subdir="comparison_baseline_explanation", cm_a=confusion_matrix, 
                     cm_b=ex_confusion_matrix, report_a=classification_report, 
                     report_b=ex_classification_report, label_a="Baseline", 
                     label_b="With Explanation")

    # Load model with masking out bullshit words fine-tuning
    mask_path = fine_tune_loop(train_df=train_df, bullshit_words=bullshit_words, fine_tuned_model_path="output/model_weights/fine_tuned_bert_masking.pth")

    # Load the fine-tuned model with masking
    tokenizer, model, device = load_fine_tuned_model(model_path=mask_path)

    # Evaluate the fine tuned model with masking
    class_report_masked, confus_matrix_masked = model_evaluation(model=model, test_df=test_df, tokenizer=tokenizer, device=device)

    # Plot the results with masking and save them in "output/plots/masking/" directory
    multiple_plots(subdir="masking", cm=confus_matrix_masked, classification_report=class_report_masked)


