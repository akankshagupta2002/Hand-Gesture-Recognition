import cv2
import os
import numpy as np
import mediapipe as mp

# ==========================================
# WEBCAM LANDMARK DATA CAPTURE
# 8 STATIC HAND GESTURES
# ==========================================

DATA_DIR = "landmark_data"
SAMPLES_PER_CLASS = 200

GESTURES = {
    "1": "01_palm",
    "2": "02_l",
    "3": "03_fist",
    "5": "05_thumb",
    "6": "06_index",
    "7": "07_ok",
    "9": "09_c",
    "0": "10_down",
}

os.makedirs(DATA_DIR, exist_ok=True)

for gesture in GESTURES.values():
    os.makedirs(
        os.path.join(DATA_DIR, gesture),
        exist_ok=True
    )

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

current_gesture = None
count = 0
started = False

print("\n" + "=" * 60)
print("WEBCAM LANDMARK DATA CAPTURE")
print("=" * 60)
print("1 = Palm")
print("2 = L")
print("3 = Fist")
print("5 = Thumb")
print("6 = Index")
print("7 = OK")
print("9 = C")
print("0 = Down")
print("Q = Quit")
print("R = Reset current gesture")
print("=" * 60)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam.")
        break

    frame = cv2.flip(frame, 1)

    display = frame.copy()

    # --------------------------------------
    # UI
    # --------------------------------------

    if current_gesture is None:

        cv2.putText(
            display,
            "Press 1,2,3,5,6,7,9 or 0",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            display,
            f"Gesture: {current_gesture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Landmarks: {count}/{SAMPLES_PER_CLASS}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        if not started:

            cv2.putText(
                display,
                "Press SPACE to start capture",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                display,
                "CAPTURING...",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

    # --------------------------------------
    # MediaPipe detection
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

        # Draw landmarks
        for point in hand:

            px = int(point.x * frame.shape[1])
            py = int(point.y * frame.shape[0])

            cv2.circle(
                display,
                (px, py),
                4,
                (0, 255, 0),
                -1
            )

        cv2.putText(
            display,
            "HAND DETECTED",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ----------------------------------
        # Capture landmarks
        # ----------------------------------

        if (
            current_gesture is not None
            and started
            and count < SAMPLES_PER_CLASS
        ):

            # Wrist = origin
            wrist_x = hand[0].x
            wrist_y = hand[0].y
            wrist_z = hand[0].z

            landmarks = []

            for point in hand:

                x = point.x - wrist_x
                y = point.y - wrist_y
                z = point.z - wrist_z

                landmarks.extend([
                    x,
                    y,
                    z
                ])

            landmarks = np.array(
                landmarks,
                dtype=np.float32
            )

            # Normalize scale
            max_value = np.max(
                np.abs(landmarks)
            )

            if max_value > 0:
                landmarks /= max_value

            folder = os.path.join(
                DATA_DIR,
                current_gesture
            )

            filename = os.path.join(
                folder,
                f"landmark_{count:04d}.npy"
            )

            np.save(
                filename,
                landmarks
            )

            count += 1

    else:

        cv2.putText(
            display,
            "NO HAND DETECTED",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # --------------------------------------
    # Show window
    # --------------------------------------

    cv2.imshow(
        "Webcam Landmark Data Capture",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Reset
    elif key == ord("r"):

        count = 0
        started = False

        print(
            f"Reset: {current_gesture}"
        )

    # Select gesture
    elif chr(key) in GESTURES:

        current_gesture = GESTURES[chr(key)]
        count = 0
        started = False

        print(
            f"\nSelected: {current_gesture}"
        )

    # Start capture
    elif key == 32:

        if current_gesture is not None:

            started = True

            print(
                f"Capturing: {current_gesture}"
            )

    # Completed
    if (
        current_gesture is not None
        and count >= SAMPLES_PER_CLASS
    ):

        print(
            f"Completed: {current_gesture} "
            f"({SAMPLES_PER_CLASS} landmarks)"
        )

        current_gesture = None
        count = 0
        started = False

# ------------------------------------------
# Cleanup
# ------------------------------------------

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("\nLandmark capture finished.")