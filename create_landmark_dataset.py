import os
import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# CREATE LANDMARK DATASET
# ==========================================

DATA_DIR = "rgb_data"
OUTPUT_FILE = "landmark_dataset.npz"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

GESTURES = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down",
]

label_map = {
    gesture: i
    for i, gesture in enumerate(GESTURES)
}

# ------------------------------------------
# MediaPipe Hand Landmarker
# ------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)

landmarker = HandLandmarker.create_from_options(
    options
)

X = []
y = []

total_images = 0
detected_images = 0

print("=" * 60)
print("CREATING MEDIAPIPE LANDMARK DATASET")
print("=" * 60)

for gesture in GESTURES:

    folder = os.path.join(
        DATA_DIR,
        gesture
    )

    print(f"\nProcessing: {gesture}")

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    class_detected = 0

    for filename in files:

        path = os.path.join(
            folder,
            filename
        )

        image = cv2.imread(path)

        if image is None:
            continue

        total_images += 1

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            continue

        hand = result.hand_landmarks[0]

        # ----------------------------------
        # Normalize landmarks
        # ----------------------------------

        landmarks = []

        # Wrist as origin
        wrist_x = hand[0].x
        wrist_y = hand[0].y
        wrist_z = hand[0].z

        for point in hand:

            x = point.x - wrist_x
            y_coord = point.y - wrist_y
            z = point.z - wrist_z

            landmarks.extend([
                x,
                y_coord,
                z
            ])

        # Normalize by maximum distance
        # from wrist
        landmarks = np.array(
            landmarks,
            dtype=np.float32
        )

        max_value = np.max(
            np.abs(landmarks)
        )

        if max_value > 0:
            landmarks = (
                landmarks / max_value
            )

        X.append(landmarks)
        y.append(label_map[gesture])

        detected_images += 1
        class_detected += 1

    print(
        f"  Images: {len(files)} | "
        f"Hands detected: {class_detected}"
    )

# ------------------------------------------
# Save dataset
# ------------------------------------------

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int32
)

np.savez(
    OUTPUT_FILE,
    X=X,
    y=y,
    class_names=np.array(GESTURES)
)

landmarker.close()

print("\n" + "=" * 60)
print("LANDMARK DATASET CREATED")
print("=" * 60)

print(f"Total images:       {total_images}")
print(f"Hands detected:     {detected_images}")
print(
    f"Detection rate:     "
    f"{detected_images / total_images * 100:.2f}%"
)

print(f"Feature shape:      {X.shape}")
print(f"Labels shape:       {y.shape}")

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)