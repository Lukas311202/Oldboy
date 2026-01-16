import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_confusion_matrix(cm, class_names, normalize=False):
    """
    Plots a confusion matrix heatmap.

    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): List of class labels
        normalize (bool): Whether to normalize the matrix
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.show()


def plot_classification_report(report_dict, class_names):
    """
    Plots precision, recall and f1-score per class.

    Args:
        report_dict (dict): classification_report(output_dict=True)
        class_names (list): Class labels
    """
    df = pd.DataFrame(report_dict).transpose()
    metrics_df = df.loc[class_names][["precision", "recall", "f1-score"]]

    combined = metrics_df.T  

    plt.figure(figsize=(8,5))
    
    bar_width = 0.35
    x = np.arange(len(combined.index)) 

    for i, cls in enumerate(class_names):
        plt.bar(x + i*bar_width, combined[cls], width=bar_width, label=cls)

    ymin = max(0, combined.min().min() - 0.02)
    ymax = min(1, combined.max().max() + 0.02)
    plt.ylim(ymin, ymax)

    plt.xticks(x + bar_width/2, combined.index)
    plt.ylabel("Score")
    plt.title("Classification Report Metrics")
    plt.legend(title="Class")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_overall_metrics(report_dict):
    """
    Plots overall model performance metrics.
    """
    overall_metrics = {
        "Accuracy": report_dict["accuracy"],
        "Macro F1": report_dict["macro avg"]["f1-score"],
        "Weighted F1": report_dict["weighted avg"]["f1-score"]
    }

    plt.figure(figsize=(6,4))
    bars = plt.bar(overall_metrics.keys(), overall_metrics.values(), color='skyblue')

    values = list(overall_metrics.values())
    ymin = max(0, min(values) - 0.01)
    ymax = min(1, max(values) + 0.01)
    plt.ylim(ymin, ymax)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.001, f"{height:.3f}", ha='center', va='bottom', fontsize=10)

    plt.ylabel("Score")
    plt.title("Overall Model Performance")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
