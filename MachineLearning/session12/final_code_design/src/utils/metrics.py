import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, f1_score, accuracy_score, 
                             classification_report)
import pandas as pd

def evaluate_and_plot(y_true, y_pred, y_proba, prefix, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{prefix} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(save_dir, f"{prefix}_cm.png"))
    plt.close()

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0,1],[0,1], linestyle='--')
        plt.title(f'{prefix} ROC (AUC={roc_auc:.3f})')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig(os.path.join(save_dir, f"{prefix}_roc.png"))
        plt.close()
    else:
        roc_auc = None

    pd.DataFrame(report).to_json(os.path.join(save_dir, "report.json"))

    return {"f1": f1, "accuracy": acc, "roc_auc": roc_auc}
