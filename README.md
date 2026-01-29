# Introduction:

This project investigates Interpretable Machine Learning (iML), also known as Explainable AI (XAI) in the field of Natural Language Processing. It is known that deep learning models for sentiment analysis possibly use tokens that correlate with labels but lack semantic meaning. With this project we aim to identify these biases and explore to which degree human intervention is useful to prevent the model to use these biases.

We utiliz a four-stage workflow:

1. **Exploration:** Fine-Tuning a BERT model on an IMDB Dataset with 50k reviews, were each review is either flagged as 'positive' or 'negative' and using Integrated Gradients (IG) to calculate attribution scores for each token.
2. **Bias Identification:** Analyzing these token statistics and isolate tokens that steer model decisions without being sentiment-relevant.
3. **Fine-Tuning adjustments:** Here we compare two strategies to remove these biases:
    - **Explanation Regularization**: We modify the loss function to penalize the model for relying on these tokens during trainings
    - **Masking:** We simply mask out these tokens in the training data.
4. **Analysis:** Compare all 3 approaches based on basic classification metrics.

## Folder structure:

The project is organized into the following structure:

    Oldboy/
    ├── data/               # Raw IMDB dataset (CSV)
    ├── output/             # Generated artifacts
    │   ├── logs/           #  JSON files with logits, stats, and attribution scores
    │   ├── model_weights/  # Model weights and training checkpoints created during execution    
    │   └── plots/          # Visualization plots
    ├── src/                # Source code
    │   ├── analysis.py     # Model analyzation and evaluation metrics
    │   ├── plotting.py     # Generating plots and comparisons
    │   ├── training.py     # Fine-tuning loops and explanation regularization
    │   └── utils.py        # Data loading, model loading, masking, and model helpers
    ├── main.py             # Main entry point to run the pipeline
    ├── requirements.txt    # List of required Python libraries
    └── README.md           # Project documentation

## Setup:
- Datensatz
- requirements.txt
- Python version ...
- Execute everything from /.
- Figure out correct pytorch version for specific gpu 

## How to use:

## Contributions:
- Dataset from [kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Pretrained Model (google-bert/bert-base-cased) from [HuggingFace](https://huggingface.co/google-bert/bert-base-cased) via the transformers library
- LLMs like Gemini and Github Copilot were used as assistence for some of the functions in the code
- [Pytorch](https://pytorch.org) library was used 
- For the word attributions calculation through Integrated Gradients, the [Captum](https://captum.ai) library was used
- Besides, the [pandas](https://pandas.pydata.org), [scikit-learn](https://scikit-learn.org/stable/), [matplotlib](https://matplotlib.org) and [seaborn](https://seaborn.pydata.org) libraries where utilized