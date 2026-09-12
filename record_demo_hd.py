import cv2
import json
import joblib
import numpy as np
import mediapipe as mp
import os
import time
from collections import deque


# ============================================================
# FULL HD HAND GESTURE DEMO RECORDER
# Same working recognition logic
# MediaPipe + Random Forest + Motion + Smoothing
# ============================================================

MODEL_PATH = "models/hand_gesture_landmark_rf.joblib"
CLASS_NAMES_PATH = "models/landmark_class_names.json"
HAND_MODEL_PATH = "hand_landmarker.task"

OUTPUT_DIR = "results/demo"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "hand_gesture_demo_HD.mp4"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

MOTION_FRAMES = 10
MOTION_THRESHOLD = 0.035
SMOOTHING_FRAMES = 5

# 60 seconds recording
MAX_DURATION = 60


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading landmark gesture model...")

model = joblib.load(
    MODEL_PATH
)

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_names = json.load(f)

print("Model loaded successfully.")
print("Classes:", class_names)


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

landmarker = (
    HandLandmarker.create_from_options(
        options
    )
)


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam not found!")
    landmarker.close()
    raise SystemExit


# Request Full HD
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1920
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    1080
)

# Request good FPS
cap.set(
    cv2.CAP_PROP_FPS,
    30
)


# Get actual camera resolution
width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

camera_fps = cap.get(
    cv2.CAP_PROP_FPS
)

if camera_fps <= 0:
    camera_fps = 30.0


print()
print("=" * 60)
print("CAMERA RESOLUTION")
print("=" * 60)
print(
    f"Actual resolution: {width} x {height}"
)
print(
    f"Camera FPS: {camera_fps:.1f}"
)
print("=" * 60)


# ============================================================
# VIDEO WRITER
# ============================================================

# Half FPS = 2x slower playback
output_fps = camera_fps / 2.0

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_FILE,
    fourcc,
    output_fps,
    (width, height)
)

if not writer.isOpened():

    print(
        "ERROR: Could not create output video."
    )

    cap.release()
    landmarker.close()

    raise SystemExit


# ============================================================
# HISTORY
# ============================================================

prediction_history = deque(
    maxlen=SMOOTHING_FRAMES
)

confidence_history = deque(
    maxlen=SMOOTHING_FRAMES
)

center_history = deque(
    maxlen=MOTION_FRAMES
)


# ============================================================
# START
# ============================================================

print()
print("=" * 60)
print("FULL HD DEMO RECORDING")
print("=" * 60)

print()
print("Recording starts automatically.")
print()
print("Show gestures clearly:")
print()
print("1. Palm")
print("2. L")
print("3. Fist")
print("4. Fist Moved")
print("5. Thumb")
print("6. Index")
print("7. OK")
print("8. Palm Moved")
print("9. C")
print("10. Down")

print()
print("Keep each gesture visible for 2-3 seconds.")
print("You can repeat gestures.")
print()
print("Maximum recording: 60 seconds")
print("Output playback: 2x slower")
print()
print("Press Q to stop early.")
print("=" * 60)


start_time = time.time()


# ============================================================
# RECORDING LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Could not read webcam frame."
        )

        break


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    elapsed = (
        time.time() - start_time
    )

    if elapsed >= MAX_DURATION:

        print()
        print(
            "60-second recording completed."
        )

        break


    # --------------------------------------------------------
    # MIRROR
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )

    display = frame.copy()

    h, w = display.shape[:2]


    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(
        mp_image
    )


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]


        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # BOX
        # ----------------------------------------------------

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


        # ----------------------------------------------------
        # LANDMARKS
        # ----------------------------------------------------

        for point in hand:

            px = int(
                point.x * w
            )

            py = int(
                point.y * h
            )

            cv2.circle(
                display,
                (px, py),
                5,
                (0, 255, 0),
                -1
            )


        cv2.putText(
            display,
            "HAND DETECTED",
            (
                x1,
                max(40, y1 - 12)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3
        )


        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------

        center_x = (
            (x1 + x2) / 2
        ) / w

        center_y = (
            (y1 + y2) / 2
        ) / h

        center_history.append(
            (center_x, center_y)
        )


        # ----------------------------------------------------
        # MOTION
        # ----------------------------------------------------

        motion_score = 0.0

        if len(center_history) >= 2:

            first_x, first_y = (
                center_history[0]
            )

            last_x, last_y = (
                center_history[-1]
            )

            motion_score = np.sqrt(
                (last_x - first_x) ** 2
                +
                (last_y - first_y) ** 2
            )

        is_moving = (
            motion_score >=
            MOTION_THRESHOLD
        )


        # ----------------------------------------------------
        # NORMALIZED LANDMARKS
        # ----------------------------------------------------

        wrist_x = hand[0].x
        wrist_y = hand[0].y
        wrist_z = hand[0].z

        landmarks = []

        for point in hand:

            x = point.x - wrist_x

            y_coord = (
                point.y - wrist_y
            )

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


        max_value = np.max(
            np.abs(landmarks)
        )

        if max_value > 0:

            landmarks /= max_value


        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        model_input = (
            landmarks.reshape(1, -1)
        )

        prediction = model.predict(
            model_input
        )[0]

        probabilities = (
            model.predict_proba(
                model_input
            )[0]
        )

        predicted_index = int(
            prediction
        )

        raw_class = (
            class_names[
                predicted_index
            ]
        )

        raw_confidence = float(
            probabilities[
                predicted_index
            ]
        )


        # ----------------------------------------------------
        # SMOOTHING
        # ----------------------------------------------------

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

        stable_class = (
            class_names[
                stable_index
            ]
        )


        # ----------------------------------------------------
        # MOTION-AWARE CLASSIFICATION
        # ----------------------------------------------------

        final_class = stable_class

        if stable_class == "01_palm":

            if is_moving:

                final_class = (
                    "08_palm_moved"
                )

            else:

                final_class = (
                    "01_palm"
                )

        elif stable_class == "03_fist":

            if is_moving:

                final_class = (
                    "04_fist_moved"
                )

            else:

                final_class = (
                    "03_fist"
                )


        # ----------------------------------------------------
        # STABLE CONFIDENCE
        # ----------------------------------------------------

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

            stable_confidence = (
                raw_confidence
            )


        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        top_indices = np.argsort(
            probabilities
        )[-3:][::-1]


        # ----------------------------------------------------
        # MAIN RESULT
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Gesture: {final_class}",
            (25, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (0, 255, 255),
            3
        )

        cv2.putText(
            display,
            (
                f"Confidence: "
                f"{stable_confidence * 100:.2f}%"
            ),
            (25, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
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
            (25, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            f"Motion score: {motion_score:.3f}",
            (25, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        y_position = 220

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
                (25, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            y_position += 32


    # ========================================================
    # NO HAND
    # ========================================================

    else:

        prediction_history.clear()
        confidence_history.clear()
        center_history.clear()

        cv2.putText(
            display,
            "NO HAND DETECTED",
            (25, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (0, 0, 255),
            3
        )

        cv2.putText(
            display,
            "Show one hand to the camera",
            (25, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )


    # ========================================================
    # RECORD
    # ========================================================

    writer.write(
        display
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Hand Gesture Recognition - FULL HD DEMO",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print()
        print(
            "Recording stopped manually."
        )

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()
cv2.destroyAllWindows()
landmarker.close()


print()
print("=" * 60)
print("FULL HD DEMO RECORDING FINISHED")
print("=" * 60)
print()
print(
    f"Video saved at:"
)
print(
    OUTPUT_FILE
)
print()
