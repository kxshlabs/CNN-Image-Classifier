import os
import sys
import ssl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

ssl._create_default_https_context = ssl._create_unverified_context

def get_model_path():
    for path in ["models/best_model.h5", "models/cnn_model.h5", "cnn_model.h5"]:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No model file found.")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    model_path = get_model_path()
    model = load_model(model_path)
    
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test_norm = x_test.astype("float32") / 255.0
    y_test_flat = y_test.flatten()
    y_test_cat = to_categorical(y_test_flat, 10)
    
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = model.predict(x_test_norm, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    loss, overall_acc = model.evaluate(x_test_norm, y_test_cat, verbose=0)
    
    cm = confusion_matrix(y_test_flat, pred_labels)
    
    plt.style.use("dark_background")
    green_cmap = LinearSegmentedColormap.from_list("green_cmap", ["#000000", "#00ff41"])
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="black")
    ax.set_facecolor("black")
    sns.heatmap(cm, annot=True, fmt="d", cmap=green_cmap, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix", color="#00ff41")
    ax.set_xlabel("Predicted Class", color="#00ff41")
    ax.set_ylabel("True Class", color="#00ff41")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    
    per_class_acc = []
    for i in range(10):
        mask = (y_test_flat == i)
        acc = np.mean(pred_labels[mask] == y_test_flat[mask]) * 100 if np.sum(mask) > 0 else 0.0
        per_class_acc.append(acc)
        
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="black")
    ax.set_facecolor("black")
    bars = ax.barh(class_names, per_class_acc, color="#00ff41")
    ax.set_xlabel("Accuracy (%)", color="#00ff41")
    ax.set_title("Per-Class Accuracy", color="#00ff41")
    ax.set_xlim(0, 100)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", color="white")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_accuracy.png"), facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    
    history_path = "models/training_history.npy"
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="black")
        ax1.set_facecolor("black")
        ax2.set_facecolor("black")
        
        ax1.plot(history["accuracy"], label="Train Acc", color="#00ff41")
        if "val_accuracy" in history:
            ax1.plot(history["val_accuracy"], label="Val Acc", color="#0080ff")
        ax1.set_title("Model Accuracy", color="#00ff41")
        ax1.set_xlabel("Epoch", color="white")
        ax1.set_ylabel("Accuracy", color="white")
        ax1.legend()
        
        ax2.plot(history["loss"], label="Train Loss", color="#00ff41")
        if "val_loss" in history:
            ax2.plot(history["val_loss"], label="Val Loss", color="#0080ff")
        ax2.set_title("Model Loss", color="#00ff41")
        ax2.set_xlabel("Epoch", color="white")
        ax2.set_ylabel("Loss", color="white")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_curves.png"), facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()

    print("┌──────────────┬──────────┐")
    print("│ Class        │ Accuracy │")
    print("├──────────────┼──────────┤")
    for name, acc in zip(class_names, per_class_acc):
        print(f"│ {name:<12} │ {acc:7.2f}% │")
    print("├──────────────┼──────────┤")
    print(f"│ Overall Loss │ {loss:8.4f} │")
    print(f"│ Overall Acc  │ {overall_acc * 100:7.2f}% │")
    print("└──────────────┴──────────┘")

if __name__ == "__main__":
    main()
