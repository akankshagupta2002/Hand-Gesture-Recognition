import cv2
import os
import time
import mediapipe as mp

# ==============================
# RGB Hand Gesture Data Capture
# ==============================

DATA_DIR = "rgb_data"
IMG_SIZE = 224
IMAGES_PER_CLASS = 200

GESTURES = {
    "1": "01_palm",
    "2": "02_l",
    "3": "03_fist",
    "4": "04_fist_moved",
    "5": "05_thumb",
    "6": "06_index",
    "7": "07_ok",
    "8": "08_palm_moved",
    "9": "09_c",
    "0": "10_down",
}

# MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)

landmarker = HandLandmarker.create_from_options(options)

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not found!")
    landmarker.close()
    exit()

print("\n======================================")
print(" RGB HAND GESTURE DATA CAPTURE")
print("======================================")
print("1 = Palm")
print("2 = L")
print("3 = Fist")
print("4 = Fist Moved")
print("5 = Thumb")
print("6 = Index")
print("7 = OK")
print("8 = Palm Moved")
print("9 = C")
print("0 = Down")
print("Q = Quit")
print("======================================\n")

current_key = None
saved_count = 0
last_save_time = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam.")
        break

    frame = cv2.flip(frame, 1)

    display = frame.copy()

    # Select gesture
    if current_key is None:
        cv2.putText(
            display,
            "Press 1-9 or 0 to select gesture",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    else:
        gesture_name = GESTURES[current_key]

        cv2.putText(
            display,
            f"Gesture: {gesture_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Images: {saved_count}/{IMAGES_PER_CLASS}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Press SPACE to start/continue | R reset | Q quit",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # MediaPipe detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        h, w = frame.shape[:2]

        x_values = [int(p.x * w) for p in hand]
        y_values = [int(p.y * h) for p in hand]

        x1 = max(0, min(x_values) - 30)
        y1 = max(0, min(y_values) - 30)
        x2 = min(w, max(x_values) + 30)
        y2 = min(h, max(y_values) + 30)

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
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Save image when space has started capture
        if (
            current_key is not None
            and saved_count < IMAGES_PER_CLASS
            and saved_count >= 0
            and time.time() - last_save_time > 0.08
        ):

            # Crop hand
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:

                crop = cv2.resize(
                    crop,
                    (IMG_SIZE, IMG_SIZE)
                )

                folder = os.path.join(
                    DATA_DIR,
                    GESTURES[current_key]
                )

                os.makedirs(folder, exist_ok=True)

                filename = os.path.join(
                    folder,
                    f"rgb_{saved_count:04d}.jpg"
                )

                cv2.imwrite(filename, crop)

                saved_count += 1
                last_save_time = time.time()

    cv2.imshow(
        "RGB Hand Gesture Data Capture",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Select gesture
    if chr(key) in GESTURES:
        current_key = chr(key)
        saved_count = 0
        last_save_time = 0

        print(
            f"\nSelected: {GESTURES[current_key]}"
        )
        print(
            f"Show this gesture in front of camera."
        )
        print(
            f"Images will automatically be saved."
        )

    # Reset current class
    elif key == ord("r"):
        saved_count = 0
        print("\nCounter reset.")

    # Space does nothing special because capture
    # automatically starts after selecting a class.
    elif key == 32:
        print("Capture active.")

    # Completed class
    if current_key is not None and saved_count >= IMAGES_PER_CLASS:
        print(
            f"\nCompleted: {GESTURES[current_key]} "
            f"({IMAGES_PER_CLASS} images)"
        )
        print("Press another number for next gesture.")

        current_key = None
        saved_count = 0


cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("\nData capture finished.")