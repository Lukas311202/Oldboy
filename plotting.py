import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

BASE_DIR = "plots/"
CLASS_NAMES = ["negative", "positive"]

def plot_confusion_matrix(cm, subdir, normalize=False):
    """
    Plots a confusion matrix heatmap.

    :param cm: Confusion matrix
    :param subdir: Subdirectory within BASE_DIR to save the plot.
    :param normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    title = "confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"
    plt.savefig(os.path.join(BASE_DIR, subdir, title), dpi=300)
    plt.close()


def plot_classification_report(report_dict, subdir):
    """
    Plots precision, recall and f1-score per class.

    :param report_dict: classification_report(output_dict=True)
    :param subdir: Subdirectory within BASE_DIR to save the plot.
    """
    df = pd.DataFrame(report_dict).transpose()
    metrics_df = df.loc[CLASS_NAMES][["precision", "recall", "f1-score"]]

    combined = metrics_df.T  

    plt.figure(figsize=(8,5))
    
    bar_width = 0.35
    x = np.arange(len(combined.index)) 

    for i, cls in enumerate(CLASS_NAMES):
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
    plt.savefig(os.path.join(BASE_DIR, subdir, "classification_report.png"), dpi=300)
    plt.close()


def plot_overall_metrics(report_dict, subdir):
    """
    Plots overall model performance metrics.

    :param report_dict: classification_report(output_dict=True)
    :param subdir: Subdirectory within BASE_DIR to save the plot.
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
    plt.savefig(os.path.join(BASE_DIR, subdir, "overall_metrics.png"), dpi=300)
    plt.close()


def plot_overall_metrics_comparison(report_a, report_b, subdir, label_a="Baseline", label_b="With Explanation"):
    """
    Compares overall metrics between two models.

    :param report_a: classification_report(output_dict=True) for model A
    :param report_b: classification_report(output_dict=True) for model B
    :param subdir: Subdirectory within BASE_DIR to save the plot.
    :param label_a: Label for model A
    :param label_b: Label for model B
    """

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
    plt.savefig(os.path.join(BASE_DIR, subdir, "overall_metrics_comparison.png"), dpi=300)
    plt.close()


def plot_classification_report_comparison(report_a, report_b, class_names, subdir, label_a="Baseline", label_b="With Explanation"):
    """
    Compares per-class precision, recall, and F1-score.

    :param report_a: classification_report(output_dict=True) for model A
    :param report_b: classification_report(output_dict=True) for model B
    :param class_names: List of class labels
    :param subdir: Subdirectory within BASE_DIR to save the plot.
    :param label_a: Label for model A
    :param label_b: Label for model B
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
        plt.savefig(os.path.join(BASE_DIR, subdir, "classification_report_comparison.png"), dpi=300)
        plt.close()


def plot_confusion_matrix_difference(cm_a, cm_b, class_names, subdir):
    """
    Plots the difference between two normalized confusion matrices.
    Positive values mean improvement in model B.

    :param cm_a: Confusion matrix of model A
    :param cm_b: Confusion matrix of model B
    :param class_names: List of class labels
    :param subdir: Subdirectory within BASE_DIR to save the plot.
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
    plt.title("Confusion Matrix Difference (Explanation - Baseline)")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, subdir, "confusion_matrix_difference.png"), dpi=300)
    plt.close()

def multiple_plots(subdir, cm, classification_report, conf_mat=True, conf_mat_norm=True, class_report=True, overall_metrics=True):
    """
    Executes multiple plotting functions and saves plots in specified subdirectory.
    On default all plot types are generated.

    :param subdir: Subdirectory within BASE_DIR to save plots.
    :param cm: Confusion matrix (np.ndarray).
    :param classification_report: Classification report dictionary.
    :param conf_mat: Whether to plot confusion matrix.
    :param conf_mat_norm: Whether to plot normalized confusion matrix.
    :param class_report: Whether to plot classification report.
    :param overall_metrics: Whether to plot overall metrics.
    """

    path = os.path.join(BASE_DIR, subdir)
    os.makedirs(path, exist_ok=True)

    if conf_mat and cm is not None:
        plot_confusion_matrix(cm, subdir, normalize=False)
    
    if conf_mat_norm and cm is not None:
        plot_confusion_matrix(cm, subdir, normalize=True)
        
    if class_report and classification_report is not None:
        plot_classification_report(classification_report, subdir)
        
    if overall_metrics and classification_report is not None:
        plot_overall_metrics(classification_report, subdir)

    print(f"All plots saved into '{path}' directory.")

def comparison_plots(subdir, cm_a, cm_b, report_a, report_b, label_a, label_b, overall_metrics=True, class_report=True, conf_mat_diff=True):
    """
    Executes multiple comparison plotting functions and saves plots in specified subdirectory.
    On default all plot types are generated.


    :param subdir: Subdirectory within BASE_DIR to save plots.
    :param cm_a: Confusion matrix of model A (np.ndarray).
    :param cm_b: Confusion matrix of model B (np.ndarray).
    :param report_a: Classification report dictionary of model A.
    :param report_b: Classification report dictionary of model B.
    :param label_a: Label for model A.
    :param label_b: Label for model B.
    :param overall_metrics: Whether to plot overall metrics comparison.
    :param class_report: Whether to plot classification report comparison.
    :param conf_mat_diff: Whether to plot confusion matrix difference.
    """

    path = os.path.join(BASE_DIR, subdir)
    os.makedirs(path, exist_ok=True)

    if overall_metrics:
        plot_overall_metrics_comparison(report_a, report_b, subdir, label_a, label_b)

    if class_report:
        plot_classification_report_comparison(report_a, report_b, CLASS_NAMES, subdir, label_a, label_b)

    if conf_mat_diff:
        plot_confusion_matrix_difference(cm_a, cm_b, CLASS_NAMES, subdir)

    print(f"All comparison plots saved into '{path}' directory.")