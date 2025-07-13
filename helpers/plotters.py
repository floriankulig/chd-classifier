import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from matplotlib.colors import ListedColormap
import numpy as np


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.ylabel("Tatsächlicher Wert")
    plt.xlabel("Vorhergesagter Wert")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(models_dict, X, y):
    plt.figure(figsize=(10, 8))
    for name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Falsch-Positiv-Rate")
    plt.ylabel("Richtig-Positiv-Rate")
    plt.title("ROC-Kurven für die besten Modelle")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_prediction_probabilities(model, X, y, model_name, ax=None):
    y_pred_proba = model.predict_proba(X)[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Getrennte Histogramme für positive und negative Klassen
    ax.hist(
        y_pred_proba[y == 0],
        bins=20,
        alpha=0.5,
        color="blue",
        label="KHK = 0 (Keine KHK)",
        density=True,
    )
    ax.hist(
        y_pred_proba[y == 1],
        bins=20,
        alpha=0.5,
        color="red",
        label="KHK = 1 (KHK vorhanden)",
        density=True,
    )

    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Vorhergesagte Wahrscheinlichkeit für KHK = 1")
    ax.set_ylabel("Dichte")
    ax.set_title(f"Verteilung der Vorhersagewahrscheinlichkeiten: {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


# Funktion zur Visualisierung der Entscheidungsgrenze
def plot_decision_boundary(model, X, y, title="Entscheidungsgrenze", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    # Erstelle ein Gitter, um die Entscheidungsgrenze zu plotten
    h = 0.02  # Schrittweite im Mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Berechne die Vorhersage für jeden Punkt im Gitter
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotte die Entscheidungsgrenze
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(["#FFAAAA", "#AAAAFF"]))

    # Plotte die Datenpunkte
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=ListedColormap(["#FF0000", "#0000FF"]),
        edgecolor="k",
        s=40,
        alpha=0.7,
    )

    ax.set_title(title)
    ax.set_xlabel("PCA Komponente 1")
    ax.set_ylabel("PCA Komponente 2")
    ax.legend(
        scatter.legend_elements()[0],
        ["KHK = 0 (Keine KHK)", "KHK = 1 (KHK vorhanden)"],
        loc="upper right",
    )

    return ax
