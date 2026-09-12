import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 6
SEED = 42

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OLD_MODEL = MODEL_DIR / "hand_gesture_mobilenetv2.keras"
NEW_MODEL = MODEL_DIR / "hand_gesture_mobilenetv2_improved.keras"

tf.keras.utils.set_random_seed(SEED)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    MODEL_DIR / "class_names.json",
    "r",
    encoding="utf-8"
) as f:
    class_names = json.load(f)


# ============================================================
# LOAD DATA
# ============================================================

def load_records(filename):
    with open(
        DATASET_DIR / filename,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


train_records = load_records("train.json")
val_records = load_records("validation.json")


train_paths = [
    item["path"] for item in train_records
]

train_labels = [
    class_names.index(item["class"])
    for item in train_records
]

val_paths = [
    item["path"] for item in val_records
]

val_labels = [
    class_names.index(item["class"])
    for item in val_records
]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_and_preprocess(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_png(
        image,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.image.grayscale_to_rgb(
        image
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = preprocess_input(image)

    return image, label


def make_dataset(paths, labels, training=False):

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        ds = ds.shuffle(
            min(len(paths), 10000),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds


train_ds = make_dataset(
    train_paths,
    train_labels,
    training=True
)

val_ds = make_dataset(
    val_paths,
    val_labels,
    training=False
)


# ============================================================
# LOAD EXISTING BEST MODEL
# ============================================================

print("=" * 60)
print("TARGETED MODEL IMPROVEMENT")
print("=" * 60)

print("\nLoading existing model...")

model = tf.keras.models.load_model(
    OLD_MODEL
)

print("Existing model loaded successfully.")


# ============================================================
# FIND MOBILENETV2 BASE
# ============================================================

base_model = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ) and "mobilenet" in layer.name.lower():

        base_model = layer
        break


if base_model is None:
    raise RuntimeError(
        "MobileNetV2 base model could not be found."
    )


print(
    f"\nBase model found: {base_model.name}"
)


# ============================================================
# FINE-TUNE MORE LAYERS
# ============================================================

base_model.trainable = True

# Freeze earlier layers.
# Fine-tune only the last 60 layers.
for layer in base_model.layers[:-60]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for stable
# transfer-learning fine-tuning.
for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False


trainable_count = sum(
    layer.trainable
    for layer in base_model.layers
)

print(
    f"Trainable MobileNetV2 layers: "
    f"{trainable_count}"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================
#
# The previous evaluation showed weak recall for:
# 07_ok
# 08_palm_moved
#
# Give these classes slightly higher importance.
# Other classes remain unchanged.
# ============================================================

class_weights = {
    0: 1.0,   # 01_palm
    1: 1.0,   # 02_l
    2: 1.0,   # 03_fist
    3: 1.0,   # 04_fist_moved
    4: 1.0,   # 05_thumb
    5: 1.0,   # 06_index
    6: 1.5,   # 07_ok
    7: 1.5,   # 08_palm_moved
    8: 1.0,   # 09_c
    9: 1.0,   # 10_down
}


print("\nClass weights:")

for i, name in enumerate(class_names):
    print(
        f"{name}: {class_weights[i]}"
    )


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=3e-6
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(NEW_MODEL),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=1,
        min_lr=1e-7,
        verbose=1
    )
]


# ============================================================
# FINE-TUNING
# ============================================================

print("\n" + "=" * 60)
print("STARTING TARGETED FINE-TUNING")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

model.save(
    NEW_MODEL
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_data = {
    key: [float(x) for x in values]
    for key, values in history.history.items()
}

with open(
    RESULTS_DIR / "improvement_history.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history_data,
        f,
        indent=4
    )


# ============================================================
# SUMMARY
# ============================================================

best_val_accuracy = max(
    history_data["val_accuracy"]
)

best_train_accuracy = max(
    history_data["accuracy"]
)

print("\n" + "=" * 60)
print("IMPROVEMENT TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best training accuracy   : "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Best validation accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print("\nImproved model saved to:")
print(NEW_MODEL)

print("\nNext step:")
print("Evaluate the improved model on the unseen test subject.")