import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


def load_and_preprocess():
    df = pd.read_csv("heart.csv")
    X = df.drop("output", axis=1)
    y = df["output"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1-score": f1
    }


def evaluate_thresholds(y_true, y_prob, thresholds=[0.3, 0.5, 0.7]):
    for thresh in thresholds:
        print(f"\n== Threshold: {thresh} ==")
        y_pred = (y_prob >= thresh).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\nAUC: {roc_auc:.4f}")


def main():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Vary thresholds
    evaluate_thresholds(y_test, y_prob, thresholds=[0.3, 0.5, 0.7])

    # Plot ROC and calculate AUC
    plot_roc(y_test, y_prob)


if __name__ == "__main__":
    main()
