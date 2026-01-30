# Introduction:

This project investigates Interpretable Machine Learning (iML), also known as Explainable AI (XAI) in the field of Natural Language Processing. It is known that deep learning models for sentiment analysis possibly use tokens that correlate with labels but lack semantic meaning. With this project we aim to identify these biases and explore to which degree human intervention is useful to prevent the model from using these biases.

We utilize a four-stage workflow:

1. **Exploration:** Fine-Tuning a BERT Model on an IMDB Dataset with 50k reviews, were each review is either flagged as 'positive' or 'negative' and using Integrated Gradients (IG) to calculate attribution scores for each token.
2. **Bias Identification:** Analyzing these token statistics and isolating tokens that steer model decisions without being sentiment-relevant.
3. **Fine-Tuning Adjustments:** Here we compare two strategies to remove these biases:
    - **Masking:** We simply mask out these tokens in the training data.
    - **Explanation Regularization**: We modify the loss function to penalize the model for relying on these tokens during training.
4. **Analysis:** Compare all 3 approaches based on basic classification metrics.

# Folder structure:

The project is organized into the following structure:

    Oldboy/
    ├── data/               # Raw IMDB dataset (CSV)
    ├── output/             # Generated artifacts
    │   ├── logs/           # JSON files with logits, classification stats, and attribution scores
    │   ├── model_weights/  # Model weights and training checkpoints created during execution    
    │   └── plots/          # Visualization plots
    ├── src/                # Source code
    │   ├── analysis.py     # Model analyzation and evaluation metrics
    |   ├── model_loops.py  # Functions for fine tuning and evaluating models
    │   ├── plotting.py     # Generating plots and comparisons
    │   ├── training.py     # Fine-tuning loops and explanation regularization
    │   └── utils.py        # Data loading, model loading, masking, and model helpers
    ├── main.py             # Main entry point to run the pipeline
    ├── requirements.txt    # List of required Python libraries
    └── README.md           # Project documentation

# Setup:
1. Install [anaconda](https://www.anaconda.com/download).
2. Create a new conda environment with Python 3.10 and activate it.
    ```
    conda create -n oldboy python=3.10
    conda activate oldboy
    ```
3. Install all requirements with `pip install -r requirements.txt`. 

# Important Notes:

- The provided PyTorch version might not work for your specific GPU. In case you recieve error messages, try to download a different PyTorch version which is compatible with your GPU.

- This workflow is highly computationally expensive and may run for a long time. Our execution with an RTX 5060 Ti with 16GB VRAM took roughly over 45 hours. However you can reduce the number of steps for the attribution calculation, but note that this will likely lead to much less representative results.

# How to use:
**Important:** Execute everything from the source folder `./`.

In order to execute the workflow with default values just run `python .\main.py`. All results are plotted and saved into `./output/plots/XXX` where 'XXX' is either `baseline`, `masking`, `with_explanations`, `comparison_baseline_explanation`, `comparison_baseline_masking` or `comparison_masking_explanation`. For raw values look into `output/logs/` where the raw results in are saved in JSON format .

If you want to understand the execution of `main.py` or tweak some parameters (for example, the batch size to be able to execute the workflow faster), the following is a brief description of the main components of the `main.py`. For each parameter or function the line is provided for easier finding:

## Setup and Data Preparation:
- `MODEL_NAME` (line 24): The Hugging Face Model ID of the sentiment analysis as a string (Default: "google-bert/bert-base-cased").
- `DATA_PATH` (line 25): The path to the dataset (Default: "data/imdb_dataset.csv").
- `original_seed` (line 26): Customize the seed.
- `repetitions` (line 28): Define how often the experiment should be executed (results are averaged).
- `test_size` (line 29): Adjust the size of the test set. 

## Default Fine-Tuning and Evaluation:
- `fine_tune_and_evaluate_model()` (line 56): The `batch_size`, `epochs`, and `learning_rate` can be adjusted. 

## Attribution Calculation:
We calculate attributions only on a subset of the training set, since we used `n_steps=500` to get representative attributions.

- `train_test_split()` (line 74): Reduces the training set. The `train_size` can be adjusted.
- `run_attributions()` (line 83): `n_steps=500` is set as default, based on your GPU this can take a long time. You can set a lower value, but this comes with a less representative result. `save_every` defines how often to make a checkpoint for the previous results. Also the `internal_batch_size` can be adjusted.

## Analyze Attributions:
Here, a list of the attribution values is generated that can be investigated by humans to identify tokens that are considered to not be biased.
 - `get_most_meaningful_words()` (line 90): Filter the `top_n` words with the highest abolute global attribution value and set a `threshold` on how often the word should at least occur.
 - `bullshit_words` (line 95): Fill the list with the words you consider to be biased.

 ## Fine-Tuning with Masking-Out:

- `fine_tune_and_evaluate_model()` (line 118): `epochs`, `batch_size,` and `learning_rate` can be tweaked.

## Fine-Tuning with Explanation-Based Loss:
Similarly, here we used `n_steps=500` to get representative attributions.

- `fine_tune_and_evaluate_model_with_explanations()` (line 135): `n_steps`, `batch_size`, `epochs`, and the lambda value `lam` to scale the explanation-based loss can be tweaked.


# Contributions:
- Dataset from [kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Pretrained Model (google-bert/bert-base-cased) from [HuggingFace](https://huggingface.co/google-bert/bert-base-cased) via the Transformers library
- For the word attributions calculation through Integrated Gradients, the [Captum](https://captum.ai) library was used
- The [Pytorch](https://pytorch.org), [pandas](https://pandas.pydata.org), [scikit-learn](https://scikit-learn.org/stable/), [matplotlib](https://matplotlib.org) and [seaborn](https://seaborn.pydata.org) libraries were used
- LLMs like Gemini and Github Copilot were used as assistance for some of the functions in the code