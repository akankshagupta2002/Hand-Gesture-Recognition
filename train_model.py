import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
NUM_CLASSES = 10

EPOCHS_FROZEN = 8
EPOCHS_FINE_TUNE = 4

SEED = 42

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

tf.keras.utils.set_random_seed(SEED)

print("=" * 60)
print("HAND GESTURE RECOGNITION - MODEL TRAINING")
print("=" * 60)

print(f"TensorFlow version: {tf.__version__}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Classes: {NUM_CLASSES}")

gpus = tf.config.list_physical_devices("GPU")
print(f"GPU devices: {gpus}")

if not gpus:
    print("Training device: CPU")


# ============================================================
# LOAD DATASET METADATA
# ============================================================

with open(DATASET_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

CLASS_NAMES = metadata["class_names"]

print("\nClasses:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name}")


# ============================================================
# LOAD SPLIT FILES
# ============================================================

def load_records(filename):
    with open(DATASET_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


train_records = load_records("train.json")
val_records = load_records("validation.json")


train_paths = [item["path"] for item in train_records]
train_labels = [
    CLASS_NAMES.index(item["class"])
    for item in train_records
]

val_paths = [item["path"] for item in val_records]
val_labels = [
    CLASS_NAMES.index(item["class"])
    for item in val_records
]

print("\nDataset:")
print(f"Training images   : {len(train_paths)}")
print(f"Validation images : {len(val_paths)}")


# ============================================================
# DATA PIPELINE
# ============================================================

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)

    # LeapGestRecog images are grayscale infrared images
    image = tf.image.decode_png(
        image,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    # Convert grayscale -> RGB for MobileNetV2
    image = tf.image.grayscale_to_rgb(image)

    image = tf.cast(image, tf.float32)

    # MobileNetV2 preprocessing: [0,255] -> [-1,1]
    image = preprocess_input(image)

    return image, label


def make_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 10000),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


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
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)


# ============================================================
# MOBILE NET V2 MODEL
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)

# Stage 1: freeze pretrained base
base_model.trainable = False


inputs = layers.Input(
    shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    name="image"
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.20)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="gesture_prediction"
)(x)


model = models.Model(
    inputs,
    outputs,
    name="hand_gesture_mobilenetv2"
)


# ============================================================
# STAGE 1 - TRAIN CLASSIFICATION HEAD
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n" + "=" * 60)
print("STAGE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 60)

model.summary()


checkpoint_path = MODEL_DIR / "hand_gesture_mobilenetv2.keras"


callbacks_stage1 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FROZEN,
    callbacks=callbacks_stage1
)


# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 60)
print("STAGE 2: FINE-TUNING MOBILE NET V2")
print("=" * 60)

# Unfreeze the base model
base_model.trainable = True

# Freeze earlier layers.
# Fine-tune only the last 30 layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


callbacks_stage2 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        mode="max",
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


history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=callbacks_stage2
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    checkpoint_path
)

print("\nModel saved to:")
print(checkpoint_path)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    MODEL_DIR / "class_names.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        CLASS_NAMES,
        f,
        indent=4
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history = {}

for key in history1.history:
    history[key] = history1.history[key].copy()


# Append Stage 2 values
for key, values in history2.history.items():

    if key in history:
        history[key].extend(values)
    else:
        history[key] = values


with open(
    RESULTS_DIR / "training_history.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# TRAINING SUMMARY
# ============================================================

best_val_accuracy = max(
    history["val_accuracy"]
)

best_train_accuracy = max(
    history["accuracy"]
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best training accuracy   : "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Best validation accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(f"\nSaved model:")
print(f"  {checkpoint_path}")

print("\nSaved files:")
print(f"  {MODEL_DIR / 'class_names.json'}")
print(f"  {RESULTS_DIR / 'training_history.json'}")

print("\nNext step: Test-set evaluation + confusion matrix.")