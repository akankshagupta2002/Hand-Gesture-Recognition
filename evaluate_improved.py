import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

DATASET_DIR = Path("dataset")
MODEL_PATH = Path("models/hand_gesture_mobilenetv2_improved.keras")
RESULTS_DIR = Path("results/improved")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL AND CLASS NAMES
# ============================================================

print("=" * 60)
print("HAND GESTURE RECOGNITION - TEST EVALUATION")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(
    "models/class_names.json",
    "r",
    encoding="utf-8"
) as f:
    class_names = json.load(f)

print(f"Model loaded: {MODEL_PATH}")
print(f"Number of classes: {len(class_names)}")


# ============================================================
# LOAD TEST DATA
# ============================================================

with open(
    DATASET_DIR / "test.json",
    "r",
    encoding="utf-8"
) as f:
    test_records = json.load(f)

test_paths = [
    item["path"]
    for item in test_records
]

test_labels = [
    class_names.index(item["class"])
    for item in test_records
]

print(f"\nTest images: {len(test_paths)}")


# ============================================================
# TEST DATA PIPELINE
# ============================================================

def load_and_preprocess_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_png(
        image,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    # Grayscale infrared -> RGB
    image = tf.image.grayscale_to_rgb(image)

    image = tf.cast(
        image,
        tf.float32
    )

    image = preprocess_input(image)

    return image, label


test_ds = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

test_ds = test_ds.map(
    load_and_preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\nEvaluating on unseen test subject...")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = np.array(
    test_labels
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\nOverall Metrics:")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

report_df.to_csv(
    RESULTS_DIR / "classification_report.csv"
)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(12, 10)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title(
    "Hand Gesture Recognition - Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PER-CLASS PERFORMANCE
# ============================================================

per_class = []

for i, class_name in enumerate(class_names):

    true_positive = cm[i, i]

    actual = cm[i, :].sum()

    predicted = cm[:, i].sum()

    class_recall = (
        true_positive / actual
        if actual > 0
        else 0
    )

    class_precision = (
        true_positive / predicted
        if predicted > 0
        else 0
    )

    if (
        class_precision + class_recall
    ) > 0:

        class_f1 = (
            2
            * class_precision
            * class_recall
            / (class_precision + class_recall)
        )

    else:
        class_f1 = 0

    per_class.append(
        {
            "class": class_name,
            "precision": class_precision,
            "recall": class_recall,
            "f1_score": class_f1,
            "support": actual,
        }
    )


per_class_df = pd.DataFrame(
    per_class
)

per_class_df.to_csv(
    RESULTS_DIR / "per_class_performance.csv",
    index=False
)


# ============================================================
# SAVE EVALUATION SUMMARY
# ============================================================

summary = {
    "test_images": len(test_paths),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "accuracy": float(accuracy),
    "weighted_precision": float(precision),
    "weighted_recall": float(recall),
    "weighted_f1_score": float(f1),
    "test_subject": "09",
    "evaluation_type": "Unseen subject test set",
}

with open(
    RESULTS_DIR / "evaluation_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("\nSaved results:")

print(
    f"  {RESULTS_DIR / 'classification_report.csv'}"
)

print(
    f"  {RESULTS_DIR / 'per_class_performance.csv'}"
)

print(
    f"  {RESULTS_DIR / 'confusion_matrix.png'}"
)

print(
    f"  {RESULTS_DIR / 'evaluation_summary.json'}"
)

print("\nTask 5 evaluation artifacts are ready.")
