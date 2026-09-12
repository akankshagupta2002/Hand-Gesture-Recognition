import os
import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# HAND LANDMARK CLASSIFIER TRAINING
# ==========================================

DATA_DIR = "landmark_data"
MODEL_DIR = "models"
RESULTS_DIR = "results/landmark"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hand_gesture_landmark_rf.joblib"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "landmark_class_names.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

GESTURES = [
    "01_palm",
    "02_l",
    "03_fist",
    "05_thumb",
    "06_index",
    "07_ok",
    "09_c",
    "10_down"
]

# ------------------------------------------
# Load landmark files
# ------------------------------------------

X = []
y = []

print("=" * 60)
print("LOADING LANDMARK DATA")
print("=" * 60)

for label, gesture in enumerate(GESTURES):

    folder = os.path.join(
        DATA_DIR,
        gesture
    )

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    ]

    print(
        f"{gesture}: {len(files)} samples"
    )

    for filename in files:

        path = os.path.join(
            folder,
            filename
        )

        landmarks = np.load(path)

        X.append(landmarks)
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("\nDataset shape:", X.shape)
print("Labels shape:", y.shape)

# ------------------------------------------
# Train / test split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------
# Random Forest
# ------------------------------------------

print("\n")
print("=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

# ------------------------------------------
# Evaluation
# ------------------------------------------

print("\n")
print("=" * 60)
print("EVALUATING LANDMARK MODEL")
print("=" * 60)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nTest Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("\nClassification Report:\n")

report = classification_report(
    y_test,
    y_pred,
    target_names=GESTURES,
    digits=4
)

print(report)

# ------------------------------------------
# Confusion matrix
# ------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# Save confusion matrix
np.savetxt(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.csv"
    ),
    cm,
    fmt="%d",
    delimiter=","
)

# ------------------------------------------
# Save classification report
# ------------------------------------------

with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(
        f"Landmark Model Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    f.write(report)

# ------------------------------------------
# Save model
# ------------------------------------------

joblib.dump(
    model,
    MODEL_PATH
)

# Save class names
with open(
    CLASS_NAMES_PATH,
    "w"
) as f:

    json.dump(
        GESTURES,
        f,
        indent=4
    )

# ------------------------------------------
# Save evaluation summary
# ------------------------------------------

summary = {
    "model": "Random Forest",
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "num_classes": len(GESTURES),
    "accuracy": float(accuracy),
    "accuracy_percent": float(
        accuracy * 100
    )
}

with open(
    os.path.join(
        RESULTS_DIR,
        "evaluation_summary.json"
    ),
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

print("\n")
print("=" * 60)
print("LANDMARK MODEL TRAINING COMPLETE")
print("=" * 60)

print("\nModel saved:")
print(MODEL_PATH)

print("\nClass names saved:")
print(CLASS_NAMES_PATH)

print("\nResults saved in:")
print(RESULTS_DIR)

print("\nFinal Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("=" * 60)