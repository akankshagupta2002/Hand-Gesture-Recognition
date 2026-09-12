import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ==========================================
# RGB HAND GESTURE MODEL TRAINING
# ==========================================

DATA_DIR = "rgb_data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hand_gesture_rgb_mobilenetv2.keras"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("RGB HAND GESTURE MODEL TRAINING")
print("=" * 60)

# ------------------------------------------
# Load dataset
# ------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print(f"\nNumber of classes: {num_classes}")

# Save class names
with open(
    os.path.join(MODEL_DIR, "rgb_class_names.json"),
    "w"
) as f:
    json.dump(class_names, f, indent=4)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ------------------------------------------
# Data augmentation
# ------------------------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.10, 0.10),
    layers.RandomContrast(0.15),
], name="data_augmentation")

# ------------------------------------------
# MobileNetV2
# ------------------------------------------

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ------------------------------------------
# Build model
# ------------------------------------------

inputs = layers.Input(
    shape=(224, 224, 3),
    name="image"
)

x = data_augmentation(inputs)

# MobileNetV2 preprocessing:
# 0-255 -> -1 to +1
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1.0,
    name="mobilenetv2_rescaling"
)(x)

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

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax",
    name="gesture_output"
)(x)

model = models.Model(
    inputs,
    outputs
)

# ------------------------------------------
# Stage 1
# ------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n")
print("=" * 60)
print("STAGE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 60)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

# ------------------------------------------
# Stage 2: Fine-tuning
# ------------------------------------------

print("\n")
print("=" * 60)
print("STAGE 2: FINE-TUNING MOBILENETV2")
print("=" * 60)

base_model.trainable = True

# Fine-tune last 40 layers
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Keep BatchNorm frozen
for layer in base_model.layers:
    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

# ------------------------------------------
# Load best model
# ------------------------------------------

print("\nLoading best saved model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

# ------------------------------------------
# Final validation
# ------------------------------------------

print("\n")
print("=" * 60)
print("FINAL RGB MODEL VALIDATION")
print("=" * 60)

loss, accuracy = best_model.evaluate(
    val_ds,
    verbose=1
)

print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Loss: {loss:.4f}")

print("\nModel saved at:")
print(MODEL_PATH)

print("\nRGB training completed successfully!")
print("=" * 60)