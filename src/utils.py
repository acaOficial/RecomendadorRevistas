# utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# ==================================================
# TEXTO: matriz bonita y legible
# ==================================================

def print_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)

    label_width = max(len(l) for l in labels) + 2
    cell_width = 7

    print("\nMatriz de Confusión (conteos)")
    print("True \\ Pred".ljust(label_width), end="")
    for l in labels:
        print(l[:cell_width].rjust(cell_width), end="")
    print()

    for i, row in enumerate(cm):
        print(labels[i].ljust(label_width), end="")
        for v in row:
            print(f"{v:>{cell_width}d}", end="")
        print()


# ==================================================
# PLOT: heatmap decente (opción normalizada)
# ==================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels,
    title="Matriz de Confusión",
    normalize=False
):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title + (" (Normalizada)" if normalize else ""))
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() * 0.5

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()
