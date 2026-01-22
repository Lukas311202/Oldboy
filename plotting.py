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


def plot_overall_metrics_comparison(report_a, report_b, label_a="Baseline", label_b="With Explanation"):
    """
    Compares overall metrics between two models.
    """
    metrics = ["accuracy", "macro avg", "weighted avg"]
    metric_names = ["Accuracy", "Macro F1", "Weighted F1"]

    values_a = [
        report_a["accuracy"],
        report_a["macro avg"]["f1-score"],
        report_a["weighted avg"]["f1-score"]
    ]

    values_b = [
        report_b["accuracy"],
        report_b["macro avg"]["f1-score"],
        report_b["weighted avg"]["f1-score"]
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, values_a, width, label=label_a)
    plt.bar(x + width/2, values_b, width, label=label_b)

    plt.xticks(x, metric_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Overall Metrics Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_classification_report_comparison(report_a, report_b, class_names, label_a="Baseline", label_b="With Explanation"):
    """
    Compares per-class precision, recall, and F1-score.
    """
    df_a = pd.DataFrame(report_a).transpose()
    df_b = pd.DataFrame(report_b).transpose()

    metrics = ["precision", "recall", "f1-score"]

    for metric in metrics:
        values_a = df_a.loc[class_names][metric]
        values_b = df_b.loc[class_names][metric]

        x = np.arange(len(class_names))
        width = 0.35

        plt.figure(figsize=(7,4))
        plt.bar(x - width/2, values_a, width, label=label_a)
        plt.bar(x + width/2, values_b, width, label=label_b)

        plt.xticks(x, class_names)
        plt.ylim(0, 1)
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} Comparison per Class")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix_difference(cm_a, cm_b, class_names):
    """
    Plots the difference between two normalized confusion matrices.
    Positive values mean improvement in model B.
    """
    cm_a = cm_a.astype("float") / cm_a.sum(axis=1, keepdims=True)
    cm_b = cm_b.astype("float") / cm_b.sum(axis=1, keepdims=True)

    diff = cm_b - cm_a

    plt.figure(figsize=(6,5))
    sns.heatmap(
        diff,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Difference (Explanation − Baseline)")
    plt.tight_layout()
    plt.show()
