# ood_metrics.py

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

# def fpr_at_95_tpr(y_true, y_score):
#     # Check if y_true and y_score have the same length
#     if len(y_true) != len(y_score):
#         raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements, y_score has {len(y_score)} elements.")

#     fpr, tpr, thresholds = roc_curve(y_true, y_score)
#     try:
#         # Ensure we find the correct threshold where TPR >= 0.95
#         return fpr[np.where(tpr >= 0.95)[0][0]]
#     except IndexError:
#         return 1.0  # Return 1.0 if no threshold with TPR >= 0.95 is found

def fpr_at_95_tpr(y_true, y_score):
    # y_true: 1 = OOD (positive), 0 = ID (negative)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Find closest TPR >= 0.95
    target_idx = np.where(tpr >= 0.95)[0]
    if len(target_idx) == 0:
        print("Warning: No TPR >= 0.95 found.")
        return 1.0  # Worst-case FPR
    idx = target_idx[0]
    return fpr[idx]


def calc_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    fpr95 = fpr_at_95_tpr(y_true, y_score)
    return auroc, auprc, fpr95

def plot_roc(y_true, y_score, filename='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(filename)

def plot_pr(y_true, y_score, filename='pr_curve.png'):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc='lower left')
    plt.savefig(filename)

def plot_barcode(y_score, filename='barcode.png'):
    plt.figure(figsize=(10, 2))
    plt.plot(y_score, np.zeros_like(y_score), '|', color='black')
    plt.title('Barcode plot')
    plt.savefig(filename)
