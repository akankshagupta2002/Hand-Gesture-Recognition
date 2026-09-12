import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (160, 160)

DATASET_DIR = Path("dataset")
MODEL_PATH = Path(
    "models/hand_gesture_mobilenetv2_improved.keras"
)
OUTPUT_DIR = Path(
    "sample_predictions/improved"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("IMPROVED MODEL - SAMPLE PREDICTIONS")
print("=" * 60)

print("\nLoading improved model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open(
    "models/class_names.json",
    "r",
    encoding="utf-8"
) as f:
    class_names = json.load(f)

with open(
    DATASET_DIR / "test.json",
    "r",
    encoding="utf-8"
) as f:
    test_records = json.load(f)

print("Improved model loaded successfully.")
print(f"Test images available: {len(test_records)}")


# ============================================================
# SELECT ONE IMAGE FROM EACH CLASS
# ============================================================

selected_records = []

for class_name in class_names:

    candidates = [
        item
        for item in test_records
        if item["class"] == class_name
    ]

    if candidates:
        selected_records.append(
            candidates[len(candidates) // 2]
        )


print(
    f"Selected samples: "
    f"{len(selected_records)}"
)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def prepare_image(image_path):

    image_bytes = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_png(
        image_bytes,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    # Grayscale infrared -> RGB
    image = tf.image.grayscale_to_rgb(
        image
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = preprocess_input(
        image
    )

    return image


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

results = []

for index, record in enumerate(
    selected_records,
    start=1
):

    image_path = record["path"]
    actual_class = record["class"]

    image = prepare_image(
        image_path
    )

    prediction = model.predict(
        tf.expand_dims(
            image,
            axis=0
        ),
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(prediction)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        prediction[predicted_index]
    )

    correct = (
        actual_class == predicted_class
    )

    # --------------------------------------------------------
    # Load original image for display
    # --------------------------------------------------------

    display_image = tf.io.read_file(
        image_path
    )

    display_image = tf.image.decode_png(
        display_image,
        channels=1
    ).numpy().squeeze()

    plt.figure(
        figsize=(7, 5)
    )

    plt.imshow(
        display_image,
        cmap="gray"
    )

    plt.axis("off")

    result_text = (
        "CORRECT"
        if correct
        else "INCORRECT"
    )

    plt.title(
        f"Actual: {actual_class}\n"
        f"Predicted: {predicted_class}\n"
        f"Confidence: {confidence * 100:.2f}%\n"
        f"Result: {result_text}"
    )

    output_path = (
        OUTPUT_DIR
        / f"prediction_{index:02d}.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    results.append(
        {
            "sample": index,
            "image_path": image_path,
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percent": (
                confidence * 100
            ),
            "correct": correct,
        }
    )

    print(
        f"{index:02d}. "
        f"Actual={actual_class} | "
        f"Predicted={predicted_class} | "
        f"Confidence={confidence * 100:.2f}% | "
        f"{result_text}"
    )


# ============================================================
# SAVE CSV
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_DIR / "sample_predictions.csv",
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

correct_count = sum(
    item["correct"]
    for item in results
)

total_count = len(results)

sample_accuracy = (
    correct_count / total_count
    if total_count > 0
    else 0
)

summary = {
    "model": str(MODEL_PATH),
    "total_samples": total_count,
    "correct_predictions": correct_count,
    "incorrect_predictions": (
        total_count - correct_count
    ),
    "sample_accuracy": sample_accuracy,
}


with open(
    OUTPUT_DIR /
    "sample_prediction_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("IMPROVED SAMPLE PREDICTIONS COMPLETED")
print("=" * 60)

print(
    f"Correct samples : "
    f"{correct_count}/{total_count}"
)

print(
    f"Sample accuracy : "
    f"{sample_accuracy * 100:.2f}%"
)

print("\nSaved files:")
print(
    f"  {OUTPUT_DIR}\\prediction_01.png"
)
print(
    f"  ... prediction_10.png"
)
print(
    f"  {OUTPUT_DIR}\\sample_predictions.csv"
)
print(
    f"  {OUTPUT_DIR}\\sample_prediction_summary.json"
)