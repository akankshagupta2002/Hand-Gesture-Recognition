import cv2
import json
import joblib
import numpy as np
import mediapipe as mp
from collections import deque

# ==========================================
# FINAL REAL-TIME HAND GESTURE RECOGNITION
# MediaPipe Landmarks + Random Forest
# + Motion Detection
# ==========================================

MODEL_PATH = "models/hand_gesture_landmark_rf.joblib"
CLASS_NAMES_PATH = "models/landmark_class_names.json"
HAND_MODEL_PATH = "hand_landmarker.task"

# Motion settings
MOTION_FRAMES = 10
MOTION_THRESHOLD = 0.035

# Prediction smoothing
SMOOTHING_FRAMES = 5

# ------------------------------------------
# Load Random Forest model
# ------------------------------------------

print("Loading landmark gesture model...")

model = joblib.load(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Model loaded successfully.")
print("Classes:", class_names)

# ------------------------------------------
# MediaPipe Hand Landmarker
# ------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)

landmarker = HandLandmarker.create_from_options(
    options
)

# ------------------------------------------
# Webcam
# ------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not found!")
    landmarker.close()
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ------------------------------------------
# Prediction / motion history
# ------------------------------------------

prediction_history = deque(
    maxlen=SMOOTHING_FRAMES
)

confidence_history = deque(
    maxlen=SMOOTHING_FRAMES
)

center_history = deque(
    maxlen=MOTION_FRAMES
)

print("\n" + "=" * 60)
print("FINAL REAL-TIME HAND GESTURE RECOGNITION")
print("=" * 60)
print("MediaPipe + Random Forest + Motion Detection")
print("Press Q to quit")
print("=" * 60)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    display = frame.copy()

    # --------------------------------------
    # MediaPipe hand detection
    # --------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        h, w = frame.shape[:2]

        # ----------------------------------
        # Bounding box
        # ----------------------------------

        x_values = [
            int(point.x * w)
            for point in hand
        ]

        y_values = [
            int(point.y * h)
            for point in hand
        ]

        padding = 35

        x1 = max(
            0,
            min(x_values) - padding
        )

        y1 = max(
            0,
            min(y_values) - padding
        )

        x2 = min(
            w,
            max(x_values) + padding
        )

        y2 = min(
            h,
            max(y_values) + padding
        )

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            "HAND DETECTED",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ----------------------------------
        # Hand center
        # ----------------------------------

        center_x = (
            (x1 + x2) / 2
        ) / w

        center_y = (
            (y1 + y2) / 2
        ) / h

        center_history.append(
            (center_x, center_y)
        )

        # ----------------------------------
        # Motion calculation
        # ----------------------------------

        motion_score = 0.0

        if len(center_history) >= 2:

            first_x, first_y = center_history[0]
            last_x, last_y = center_history[-1]

            motion_score = np.sqrt(
                (last_x - first_x) ** 2
                +
                (last_y - first_y) ** 2
            )

        is_moving = (
            motion_score >= MOTION_THRESHOLD
        )

        # ----------------------------------
        # Create normalized 21 landmarks
        # ----------------------------------

        wrist_x = hand[0].x
        wrist_y = hand[0].y
        wrist_z = hand[0].z

        landmarks = []

        for point in hand:

            x = point.x - wrist_x
            y_coord = point.y - wrist_y
            z = point.z - wrist_z

            landmarks.extend([
                x,
                y_coord,
                z
            ])

        landmarks = np.array(
            landmarks,
            dtype=np.float32
        )

        # Same normalization used during training
        max_value = np.max(
            np.abs(landmarks)
        )

        if max_value > 0:
            landmarks /= max_value

        # ----------------------------------
        # Random Forest prediction
        # ----------------------------------

        prediction = model.predict(
            landmarks.reshape(1, -1)
        )[0]

        probabilities = model.predict_proba(
            landmarks.reshape(1, -1)
        )[0]

        predicted_index = int(prediction)

        raw_class = class_names[
            predicted_index
        ]

        raw_confidence = float(
            probabilities[predicted_index]
        )

        # ----------------------------------
        # Prediction smoothing
        # ----------------------------------

        prediction_history.append(
            predicted_index
        )

        confidence_history.append(
            raw_confidence
        )

        counts = np.bincount(
            prediction_history,
            minlength=len(class_names)
        )

        stable_index = int(
            np.argmax(counts)
        )

        stable_class = class_names[
            stable_index
        ]

        # ----------------------------------
        # Motion-aware final classification
        # ----------------------------------

        final_class = stable_class

        # Palm / Palm Moved
        if stable_class == "01_palm":

            if is_moving:
                final_class = "08_palm_moved"
            else:
                final_class = "01_palm"

        # Fist / Fist Moved
        elif stable_class == "03_fist":

            if is_moving:
                final_class = "04_fist_moved"
            else:
                final_class = "03_fist"

        # ----------------------------------
        # Confidence
        # ----------------------------------

        stable_confidences = []

        for idx, conf in zip(
            prediction_history,
            confidence_history
        ):

            if idx == stable_index:
                stable_confidences.append(
                    conf
                )

        if stable_confidences:

            stable_confidence = (
                sum(stable_confidences)
                /
                len(stable_confidences)
            )

        else:

            stable_confidence = raw_confidence

        # ----------------------------------
        # Top 3 predictions
        # ----------------------------------

        top_indices = np.argsort(
            probabilities
        )[-3:][::-1]

        # ----------------------------------
        # Display final result
        # ----------------------------------

        cv2.putText(
            display,
            f"Gesture: {final_class}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Confidence: {stable_confidence * 100:.2f}%",
            (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        motion_text = (
            "MOVING"
            if is_moving
            else "STATIC"
        )

        cv2.putText(
            display,
            f"Motion: {motion_text}",
            (20, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Motion score: {motion_score:.3f}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ----------------------------------
        # Top 3
        # ----------------------------------

        y_position = 190

        for rank, idx in enumerate(
            top_indices,
            start=1
        ):

            text = (
                f"{rank}. "
                f"{class_names[idx]} "
                f"{probabilities[idx] * 100:.1f}%"
            )

            cv2.putText(
                display,
                text,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            y_position += 28

    else:

        # ----------------------------------
        # No hand
        # ----------------------------------

        prediction_history.clear()
        confidence_history.clear()
        center_history.clear()

        cv2.putText(
            display,
            "NO HAND DETECTED",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        cv2.putText(
            display,
            "Show one hand to the camera",
            (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    # --------------------------------------
    # Show
    # --------------------------------------

    cv2.imshow(
        "Final Real-Time Hand Gesture Recognition",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ------------------------------------------
# Cleanup
# ------------------------------------------

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("\nWebcam test finished.")