import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, mean_absolute_error
import re

def plot_loss_curves_from_logfile(txt_path, save_path=None):
    pattern = re.compile(r"E(\d+)\s+tr_loss=([0-9.]+)\s+val_mae=([0-9.]+)")

    epochs = []
    tr_losses = []
    val_maes = []

    with open(txt_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                tr_losses.append(float(match.group(2)))
                val_maes.append(float(match.group(3)))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, tr_losses, label='Train Loss')
    plt.plot(epochs, val_maes, label='Val MAE')
    plt.xlabel("Epoch")
    plt.ylabel("Loss / MAE")
    plt.title("Training Loss & Validation MAE over Epochs", fontsize=13, weight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_and_save_classification_results(csv_path, save_path=None):
    df = pd.read_csv(csv_path)

    # Support either legacy 'patient_ID' or new 'video' id column.
    id_col = "video" if "video" in df.columns else ("patient_ID" if "patient_ID" in df.columns else df.columns[0])
    df[id_col] = df[id_col].astype(str)

    def pdff_to_binary(val):
        return 0 if val <= 5 else 1

    def pdff_to_class(val):
        if val < 6.4:
            return 0
        elif val < 16.3:
            return 1
        elif val < 20.7:
            return 2
        else:
            return 3

    df['Binary_GT'] = df['target'].apply(pdff_to_binary)
    df['Binary_Pred'] = df['prediction'].apply(pdff_to_binary)
    df['Class_GT'] = df['target'].apply(pdff_to_class)
    df['Class_Pred'] = df['prediction'].apply(pdff_to_class)

    # Calculate metrics
    f1_binary = f1_score(df['Binary_GT'], df['Binary_Pred'])
    mae_binary = mean_absolute_error(df['target'], df['prediction'])

    f1_multi = f1_score(df['Class_GT'], df['Class_Pred'], average='weighted')
    mae_multi = mean_absolute_error(df['target'], df['prediction'])


    fig, axs = plt.subplots(2, 2, figsize=(18, 10))

    # Top-left: Binary classification
    mismatched_binary = df[df['Binary_GT'] != df['Binary_Pred']]
    axs[0, 0].plot(df[id_col], df['target'], label='GroundTruth', marker='o', color="#6ed659", markersize=5)
    axs[0, 0].plot(df[id_col], df['prediction'], label='Predicted', marker='x', color="#4c8ebd", markersize=5)
    axs[0, 0].scatter(mismatched_binary[id_col], mismatched_binary['prediction'], color='crimson', label='Misclassified', zorder=5, s=40)
    axs[0, 0].axhline(y=5, color='red', linestyle='--', label='Threshold = 5%')
    axs[0, 0].set_title(f"Binary Classification, MAE: {mae_binary:.2f}", fontsize=12, weight='bold')
    axs[0, 0].set_ylabel("PDFF (%)")
    axs[0, 0].set_xticks(range(len(df[id_col])))
    axs[0, 0].set_xticklabels(df[id_col], rotation=90, fontsize=6)
    axs[0, 0].legend(fontsize=9)
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    # Top-right: Binary confusion matrix
    cm_binary = confusion_matrix(df['Binary_GT'], df['Binary_Pred'], labels=[0, 1])
    disp_binary = ConfusionMatrixDisplay(confusion_matrix=cm_binary, display_labels=["Class 0", "Class 1"])
    disp_binary.plot(ax=axs[0, 1], cmap='YlGnBu', colorbar=False, values_format='d')
    axs[0, 1].set_title(f"Confusion Matrix (Binary)\nF1 Score: {f1_binary:.2f}", fontsize=12)

    # Bottom-left: 4-Class classification
    mismatched_multi = df[df['Class_GT'] != df['Class_Pred']]
    axs[1, 0].plot(df[id_col], df['target'], label='GroundTruth', marker='o', color="#6ed659", markersize=5)
    axs[1, 0].plot(df[id_col], df['prediction'], label='Predicted', marker='x', color="#4c8ebd", markersize=5)
    axs[1, 0].scatter(mismatched_multi[id_col], mismatched_multi['prediction'], color='crimson', label='Misclassified', zorder=5, s=40)
    for thresh in [6.4, 16.3, 20.7]:
        axs[1, 0].axhline(y=thresh, color='red', linestyle='--')
    axs[1, 0].set_title(f"4-Class Classification, MAE: {mae_multi:.2f}", fontsize=12, weight='bold')
    axs[1, 0].set_xlabel("Patient ID")
    axs[1, 0].set_ylabel("PDFF (%)")
    axs[1, 0].set_xticks(range(len(df[id_col])))
    axs[1, 0].set_xticklabels(df[id_col], rotation=90, fontsize=6)
    axs[1, 0].legend(fontsize=9)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    # Bottom-right: 4-class confusion matrix
    cm_multi = confusion_matrix(df['Class_GT'], df['Class_Pred'], labels=[0, 1, 2, 3])
    disp_multi = ConfusionMatrixDisplay(confusion_matrix=cm_multi, display_labels=["Class 0", "Class 1", "Class 2", "Class 3"])
    disp_multi.plot(ax=axs[1, 1], cmap='YlGnBu', colorbar=False, values_format='d')
    axs[1, 1].set_title(f"Confusion Matrix (4-Class)\nF1 Score: {f1_multi:.2f}", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()