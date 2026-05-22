import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd

from q2_train_models import (
    DATA_DIR,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    NUM_CLASSES,
    DEVICE,
    CIFAR_MEAN,
    CIFAR_STD,
    DISPLAY_NAMES,
    MODEL_NAMES,
    build_model,
)


BATCH_SIZE = 128
ALL_MODELS = ["custom_cnn"] + MODEL_NAMES


def make_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "metrics"), exist_ok=True)


def build_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    test_set = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=transform)
    return DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())


def load_trained_model(name):
    model = build_model(name).to(DEVICE)
    checkpoint = os.path.join(CHECKPOINT_DIR, f"{name}_best.pth")
    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state)
    return model


def evaluate_model(model, loader):
    model.eval()
    predictions = []
    labels_all = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            labels_all.extend(labels.numpy())

    y_true = np.array(labels_all)
    y_pred = np.array(predictions)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    tp_per_class = np.diag(cm)
    fp_per_class = cm.sum(axis=0) - tp_per_class
    fn_per_class = cm.sum(axis=1) - tp_per_class
    tn_per_class = cm.sum() - (tp_per_class + fp_per_class + fn_per_class)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "TP": int(tp_per_class.sum()),
        "TN": int(tn_per_class.sum()),
        "FP": int(fp_per_class.sum()),
        "FN": int(fn_per_class.sum()),
    }

    return metrics, y_true, y_pred


def plot_combined_history(histories):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Training and Validation Curves", fontsize=15)

    items = [
        ("train_loss", "Training Loss", "Loss"),
        ("val_loss", "Validation Loss", "Loss"),
        ("train_acc", "Training Accuracy", "Accuracy (%)"),
        ("val_acc", "Validation Accuracy", "Accuracy (%)"),
    ]

    for ax, (key, title, ylabel) in zip(axes.ravel(), items):
        for name, hist in histories.items():
            values = hist[key]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, label=DISPLAY_NAMES.get(name, name), linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "q2_training_curves_all_models.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison(results):
    names = list(results.keys())
    labels = [DISPLAY_NAMES.get(name, name) for name in names]
    accuracies = [results[name]["accuracy"] * 100 for name in names]
    f1_scores = [results[name]["f1_macro"] * 100 for name in names]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, accuracies)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 100)
    for bar, value in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "q2_accuracy_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, f1_scores)
    plt.ylabel("Macro F1-score (%)")
    plt.title("Model F1-score Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 100)
    for bar, value in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "q2_f1_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_preview(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_preview = cm[:20, :20]

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_preview, interpolation="nearest", aspect="auto")
    plt.colorbar()
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion Matrix Preview: First 20 Classes")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "q2_confusion_matrix_preview.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_results_table(results):
    rows = []
    for name, values in results.items():
        rows.append({
            "Model": DISPLAY_NAMES.get(name, name),
            "Accuracy": values["accuracy"],
            "TP": values["TP"],
            "TN": values["TN"],
            "FP": values["FP"],
            "FN": values["FN"],
            "Macro F1-score": values["f1_macro"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "metrics", "evaluation_results.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "metrics", "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(df.to_string(index=False))


def main():
    make_dirs()
    test_loader = build_test_loader()
    results = {}
    preview_labels = None
    preview_preds = None

    for name in ALL_MODELS:
        print(f"Evaluating {DISPLAY_NAMES.get(name, name)}")
        model = load_trained_model(name)
        metrics, y_true, y_pred = evaluate_model(model, test_loader)
        results[name] = metrics

        if name == "custom_cnn":
            preview_labels = y_true
            preview_preds = y_pred

    histories_path = os.path.join(OUTPUT_DIR, "metrics", "training_histories.json")
    if os.path.exists(histories_path):
        with open(histories_path, "r") as f:
            histories = json.load(f)
        plot_combined_history(histories)

    plot_comparison(results)
    if preview_labels is not None:
        plot_confusion_matrix_preview(preview_labels, preview_preds)

    save_results_table(results)
    print("Evaluation files saved in outputs/metrics and outputs/plots.")


if __name__ == "__main__":
    main()
