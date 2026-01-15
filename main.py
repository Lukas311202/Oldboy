from analysis import model_evaluation
from model_loops import fine_tune_loop, run_attributions
from torch.optim import AdamW 
import torch
from utils import create_train_test_split, load_fine_tuned_model, load_fine_tuned_model

if __name__ == "__main__":
    
    print("torch version:", torch.__version__)
    
    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    else:
        print("CUDA not found. Check your drivers.")
    
    MODEL_NAME = "google-bert/bert-base-cased"

    # Train-test split
    train_df, test_df = create_train_test_split(data="imdb_dataset.csv", text_column="review", label_column="sentiment",
                                                test_size=0.2, seed=42, stratify=True)
    
    # Fine-tune the model
    fine_tuned_model_path = fine_tune_loop(train_df=train_df, base_model=MODEL_NAME, epochs=3, batch_size=16, learning_rate=2e-5)

    # Load the fine-tuned model 
    tokenizer, fine_tuned_model, device = load_fine_tuned_model(model_name=MODEL_NAME, model_path=fine_tuned_model_path)
    
    # Evaluate
    model_evaluation(model=fine_tuned_model, test_df=test_df, tokenizer=tokenizer, device=device)

    """
    Further Fine-tuned model evaluations (plotting etc.)
    """

    # Calculate word attributions
    attribution_values_json = run_attributions(tokenizer=tokenizer, model=fine_tuned_model, train_df=test_df)

    """
    Council decision, which words to consider as bullshit words
    """



    





