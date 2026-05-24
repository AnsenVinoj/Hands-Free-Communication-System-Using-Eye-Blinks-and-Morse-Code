"""
YOLOv8 best.pt Evaluation Script
---------------------------------
Evaluates trained YOLOv8 model and plots accuracy metrics.
Works with your current folder structure (mores dataset).
"""

import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ────────────────────────────────────────────────
# SETTINGS (Update if needed)
# ────────────────────────────────────────────────
MODEL_PATH = "best.pt"
DATA_YAML = "mores/data.yaml"
OUTPUT_IMG = "yolo_best_accuracy.png"


def evaluate_and_plot():
    print("\n🚀 Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)

    print("📊 Running validation...")
    metrics = model.val(data=DATA_YAML)

    # Extract detection metrics
    map50 = metrics.box.map50
    map5095 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr

    print("\n📈 Evaluation Results")
    print(f"mAP@0.5        : {map50:.4f}")
    print(f"mAP@0.5:0.95   : {map5095:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")

    # ────────────────────────────────────────────────
    # Plot Accuracy Bar Graph
    # ────────────────────────────────────────────────
    labels = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]
    values = [map50, map5095, precision, recall]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)

    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("YOLOv8 Evaluation Metrics (best.pt)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha="center"
        )

    plt.gcf().patch.set_facecolor("white")
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\n✅ Accuracy plot saved as → {OUTPUT_IMG}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
    elif not os.path.exists(DATA_YAML):
        print(f"❌ data.yaml not found: {DATA_YAML}")
    else:
        evaluate_and_plot()