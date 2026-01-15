from model_loops import fine_tune_loop
from torch.optim import AdamW 
import torch
from utils import create_train_test_split

if __name__ == "__main__":
    
    print("torch version:", torch.__version__)
    
    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    else:
        print("CUDA not found. Check your drivers.")

    # Train-test split
    train_df, test_df = create_train_test_split(data="imdb_dataset.csv", text_column="review", label_column="sentiment",
                                                test_size=0.2, seed=42, stratify=True)
    
    # Fine-tune the model
    fine_tuned_model_path = fine_tune_loop(train_df=train_df)

    


