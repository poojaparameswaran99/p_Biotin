import pandas as pd
import numpy as np 
import os 
import sys
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, auc, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


#### input as torch tensors.
def compute(ytrue, ypred, threshold, method='train'):
    acc = compute_acc(ytrue, ypred, threshold)
    auroc = compute_auroc(ytrue, ypred)
    CM = compute_CM(ytrue, ypred, threshold)
    aupr = compute_aupr(ytrue, ypred, threshold)
    recall = compute_recall(ytrue, ypred, threshold)
    precision = compute_precision(ytrue, ypred, threshold)
    if method.lower() in ['validation', 'test', 'inference']:
        barplot = make_metric_barplot(acc, auroc, aupr, recall, precision,
                                          title=f"{method} Metrics")
        heatmap = plot_cm_with_totals(CM) ## n, p correct?
    return acc, auroc, aupr, CM, recall, precision, barplot, heatmap

def compute_acc(ytrue, ypred, threshold):
    ypred = (ypred >threshold).float()
    matches = (ypred == ytrue)
    acc = torch.sum(matches) / len(ypred)
    return acc

def compute_auroc(ytrue, ypred):
    ## rocauc requires raw probs
    return roc_auc_score(ytrue, ypred)

def compute_CM(ytrue, ypred, threshold):
    ypred = (ypred >threshold).float()
    ## tn, fp, fn, tp 
    cm = confusion_matrix(ytrue, ypred,labels=[0,1])
    return cm

def compute_aupr(ytrue, ypred, threshold):
    ypred = (ypred >threshold).float()
    return average_precision_score(ytrue, ypred)

def compute_recall(ytrue, ypred, threshold):
    ## tp / (tp + fn)
    ypred = (ypred > threshold).float()
    tp = ((ytrue == 1) & (ypred == 1)).sum()
    fn = ((ytrue ==1) & (ypred == 0)).sum()
    return tp /(tp + fn)

def compute_precision(ytrue, ypred, threshold):
    ## tp / (tp + fp)
    ypred = (ypred > threshold).float()
    tp = ((ytrue == 1) & (ypred == 1)).sum()
    fp = ((ytrue == 0) & (ypred ==1)).sum()
    return tp / (tp + fp)

def make_metric_barplot(acc, auroc, aupr, recall, precision, title="Performance Metrics"):
    metrics = ["Accuracy", "AUROC", "AUPR", "Recall", "Precision"]
    values  = [acc, auroc, aupr, recall, precision]
    print(values)

    # Convert all tensors to floats
    vals = [v.item() if hasattr(v, "item") else float(v) for v in values]

    barplot, ax = plt.subplots(figsize=(6,4))

    bars = ax.bar(metrics, vals)
    ax.set_ylim(0, 1)                     # all metrics in [0,1]
    ax.set_title(title)
    ax.set_ylabel("Score")

    # Label bars with values
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    return barplot


def plot_cm_with_totals(cm, title="New Confusion Matrix"):
    """
    cm = [[TN, FP],
          [FN, TP]]
    """
    cm = np.array(cm)

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["Neg", "Pos"],
        yticklabels=["True", "False"],
        ax=ax
    )

    ax.set_title(title)

    fig.tight_layout()
    return fig
