import json
from src.analysis import average_jsonl_results, model_evaluation, get_most_meaningful_words
from src.training import fine_tune_loop, run_attributions, fine_tune_with_explanations
from src.model_loops import fine_tune_and_evaluate_model, fine_tune_and_evaluate_model_with_explanations
import torch
from src.utils import create_train_test_split, load_fine_tuned_model, load_fine_tuned_model, set_seed
from src.plotting import (
    comparison_plots,
    multiple_plots,
)
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    print("torch version:", torch.__version__)
    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    else:
        print("CUDA not found. Check your drivers.")

    """Setup and Data Preparation"""
    
    MODEL_NAME = "google-bert/bert-base-cased"
    DATA_PATH = "data/imdb_dataset.csv"
    original_seed = 42 # Seed for reproducibility
    
    loop_seed = original_seed # Temporary seed to increment during loops
    repetitions = 5 # Number of repetitions for experiments

    # Set paths for saving results
    default_results_path = "output/logs/fine_tuned_bert_results.jsonl"
    masked_results_path = "output/logs/fine_tuned_bert_masking_results.jsonl"
    explanation_results_path = "output/logs/fine_tuned_bert_explanation_results.jsonl"

    model_save_path = f"output/model_weights/fine_tuned_bert_seed_{original_seed}.pth"

    """Default Fine-tuning and Evaluation"""

    # Create train-test split once with original seed
    train_df, test_df = create_train_test_split(data=DATA_PATH, label_column="sentiment", 
                                                test_size=0.2, seed=original_seed, stratify=True)

    for _ in range(repetitions):

        # Set seed for reproducibility
        set_seed(loop_seed)

        # Set path for saving model
        current_model_save_path = f"output/model_weights/fine_tuned_bert_seed_{loop_seed}.pth"
        
        # Train-test split
        current_train_df, current_test_df = create_train_test_split(data=DATA_PATH, label_column="sentiment",
                                                test_size=0.2, seed=loop_seed, stratify=True)

        default_fine_tuning_results = fine_tune_and_evaluate_model(train_df=current_train_df, test_df=current_test_df, model_id=MODEL_NAME, seed=loop_seed,
                                                                    epochs=3, batch_size=16, learning_rate=2e-5,
                                                                    model_save_path=current_model_save_path,
                                                                    result_save_path=default_results_path, bullshit_words=None)

        loop_seed += 1

    # Average results over repetitions 
    avg_cr, avg_cm = average_jsonl_results(default_results_path, "output/logs/fine_tuned_bert_average_results.json")
        
    # Plot the baseline results and save them in "output/plots/baseline/" directory
    multiple_plots(subdir="baseline", cm=avg_cm, classification_report=avg_cr)

    """Attribution Calculation"""

    set_seed(original_seed)

    # Reduce dataset for attribution calculations
    reduced_df, _ = train_test_split(
        train_df, 
        train_size=10000, 
        stratify=train_df['sentiment'], 
        random_state=original_seed
    )

    tokenizer, model, device = load_fine_tuned_model(model_name=MODEL_NAME, model_path=model_save_path)

    # Calculate word attributions
    attribution_values_json, final_avg_delta = run_attributions(n_steps=500, save_every=15, internal_batch_size=16, 
                                                                tokenizer=tokenizer, model=model_save_path, train_df=reduced_df) 

    """Analyze Attributions"""

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

    """Fine-Tuning with Masking-Out"""

    loop_seed = original_seed

    # Repeat experiments with different seeds
    for _ in range(repetitions):
        # Set seed for reproducibility
        set_seed(loop_seed)

        # Set path for saving model
        current_model_save_path = f"output/model_weights/fine_tuned_bert_masking_seed_{loop_seed}.pth"
        
        # Train-test split
        current_train_df, current_test_df = create_train_test_split(data=DATA_PATH, label_column="sentiment",
                                                test_size=0.2, seed=loop_seed, stratify=True)

        default_fine_tuning_results = fine_tune_and_evaluate_model(train_df=current_train_df, test_df=current_test_df, model_id=MODEL_NAME, seed=loop_seed,
                                                                    epochs=3, batch_size=16, learning_rate=2e-5,
                                                                    model_save_path=current_model_save_path,
                                                                    result_save_path=masked_results_path, bullshit_words=bullshit_words) # Now with bullshit words

        loop_seed += 1

    # Average results over repetitions 
    avg_cr, avg_cm = average_jsonl_results(masked_results_path, "output/logs/fine_tuned_bert_masking_average_results.json")
        
    # Plot the masking results and save them in "output/plots/masking/" directory
    multiple_plots(subdir="masking", cm=avg_cm, classification_report=avg_cr)

    """Fine-Tuning with Explanation-Based Loss"""

    set_seed(original_seed)

    fine_tune_and_evaluate_model_with_explanations(train_df=train_df, test_df=test_df, model_name=MODEL_NAME, n_steps=500, batch_size=80, 
                                                   epochs=3, bullshit_words=bullshit_words, checkpoint_every_n_step=5, 
                                                   lam=1.0, fine_tuned_model_path="output/model_weights/fine_tuned_bert_with_ex.pth", 
                                                   tokenizer=tokenizer, device=device, seed=original_seed, result_save_path=explanation_results_path)
    
    avg_cr, avg_cm = average_jsonl_results(explanation_results_path, "output/logs/fine_tuned_bert_explanation_average_results.json")

    # Plot the results with explanations and save them in "output/plots/with_explanations/" directory
    multiple_plots(subdir="with_explanations", cm=avg_cm, classification_report=avg_cr)
 
    # Comparison plots between baseline and explanation-based model
    # comparison_plots(subdir="comparison_baseline_explanation", cm_a=confusion_matrix, 
    #                  cm_b=ex_confusion_matrix, report_a=classification_report, 
    #                  report_b=ex_classification_report, label_a="Baseline", 
    #                  label_b="With Explanation")



