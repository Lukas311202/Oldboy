import json
from src.analysis import model_evaluation
from src.training import fine_tune_loop, fine_tune_with_explanations
from src.utils import load_fine_tuned_model, load_fine_tuned_model, set_seed


def fine_tune_and_evaluate_model(train_df, test_df, model_id, seed, epochs, 
                                 batch_size, learning_rate,  model_save_path, result_save_path, bullshit_words):
    """
    Fine-tunes a model and evaluates it on the test dataset.
    Results are saved to a specified JSONL file.

    :param train_df: DataFrame containing training data.
    :param test_df: DataFrame containing testing data.
    :param model_id: HuggingFace identifier for the base model to be fine-tuned.
    :param seed: Random seed for reproducibility.
    :param epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :param learning_rate: Learning rate for the optimizer.
    :param model_save_path: Path to save the fine-tuned model.
    :param result_save_path: Path to save the evaluation results.
    :param bullshit_words: List of words to be masked during fine-tuning. Leave None for no masking.
    """
        
    # Fine-tune the model
    fine_tuned_model_path = fine_tune_loop(train_df=train_df, base_model=model_id, 
                                        fine_tuned_model_path=f"{model_save_path}.pth", 
                                        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                        bullshit_words=bullshit_words)
    
    # Load the fine-tuned model 
    tokenizer, fine_tuned_model, device = load_fine_tuned_model(model_name=model_id, model_path=fine_tuned_model_path)

    # Evaluate fine-tuned model
    classification_report, confusion_matrix = model_evaluation(model=fine_tuned_model, test_df=test_df, tokenizer=tokenizer, device=device)

    # Bind results to a dictionary
    results_to_save = {
        "seed": seed,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix.tolist()
    }
    
    # Write results to a JSONL file
    with open(result_save_path, "a") as f:
        f.write(json.dumps(results_to_save) + "\n")

def fine_tune_and_evaluate_model_with_explanations(train_df, test_df, model_name, n_steps, batch_size, 
                                                   epochs, bullshit_words, checkpoint_every_n_step, 
                                                   lam, fine_tuned_model_path, tokenizer, device, seed, result_save_path):
    """
    Fine-tunes a model with explanation-based loss and evaluates it on the test dataset.
    Reults are saved to a specified JSONL file.

    :param train_df: DataFrame containing training data.
    :param test_df: DataFrame containing testing data.
    :param model_name: HuggingFace identifier for the base model to be fine-tuned
    :param n_steps: Number of steps for attribution computation.
    :param batch_size: Batch size for training.
    :param epochs: Number of training epochs.
    :param bullshit_words: List of words to consider for the explanation-based loss.
    :param checkpoint_every_n_step: Frequency of checkpoints during fine-tuning.
    :param lam: Lambda parameter for weighting the explanation-based loss.
    :param fine_tuned_model_path: Path to save the fine-tuned model.
    :param tokenizer: Tokenizer corresponding to the model.
    :param device: Device to run the model on (CPU or GPU).
    :param seed: Random seed for reproducibility.
    :param result_save_path: Path to save the evaluation results.
    """

    # Fine-tune again with explanation-based loss
    ex_model_path = fine_tune_with_explanations(train_df, 
                                 n_steps=n_steps, 
                                 batch_size=batch_size, 
                                 epochs=epochs,
                                 bullshit_words=bullshit_words,
                                 checkpoint_every_n_step=checkpoint_every_n_step,
                                 lam=lam,
                                 fine_tuned_model_path=fine_tuned_model_path
                                )

    # Load the fine-tuned model with explanations
    _, ex_model, _ = load_fine_tuned_model(
        model_name=model_name,
        model_path=ex_model_path
    )

    # Evaluate the fine tuned model with explanations
    ex_classification_report, ex_confusion_matrix = model_evaluation( model=ex_model, test_df=test_df, tokenizer=tokenizer, device=device)

    results_to_save = {
        "seed": seed,
        "classification_report": ex_classification_report,
        "confusion_matrix": ex_confusion_matrix.tolist()
    }
    
    # Write results to a JSONL file
    with open(result_save_path, "a") as f:
        f.write(json.dumps(results_to_save) + "\n")