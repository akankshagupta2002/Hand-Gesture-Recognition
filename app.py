import json
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

LANDMARK_MODEL_PATH = Path(
    "models/hand_gesture_landmark_rf.joblib"
)

LANDMARK_CLASSES_PATH = Path(
    "models/landmark_class_names.json"
)

HAND_LANDMARKER_PATH = Path(
    "hand_landmarker.task"
)

# Original Task 5 model is kept as an optional IR fallback
IR_MODEL_PATH = Path(
    "models/hand_gesture_mobilenetv2_improved.keras"
)

IR_CLASS_NAMES_PATH = Path(
    "models/class_names.json"
)

IR_IMG_SIZE = (160, 160)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="✋",
    layout="centered"
)


# ============================================================
# LOAD LANDMARK MODEL
# ============================================================

@st.cache_resource
def load_landmark_model():

    return joblib.load(
        LANDMARK_MODEL_PATH
    )


@st.cache_data
def load_landmark_class_names():

    with open(
        LANDMARK_CLASSES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


landmark_model = load_landmark_model()

landmark_class_names = (
    load_landmark_class_names()
)


# ============================================================
# LOAD ORIGINAL IR MODEL
# ============================================================

@st.cache_resource
def load_ir_model():

    if not IR_MODEL_PATH.exists():
        return None

    try:
        import tensorflow as tf

        return tf.keras.models.load_model(
            IR_MODEL_PATH
        )

    except Exception:
        return None


@st.cache_data
def load_ir_class_names():

    if not IR_CLASS_NAMES_PATH.exists():
        return []

    with open(
        IR_CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


ir_model = load_ir_model()
ir_class_names = load_ir_class_names()


# ============================================================
# LOAD MEDIAPIPE HAND LANDMARKER
# ============================================================

@st.cache_resource
def load_hand_detector():

    base_options = (
        mp.tasks.BaseOptions(
            model_asset_path=str(
                HAND_LANDMARKER_PATH
            )
        )
    )

    options = (
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                mp.tasks.vision.RunningMode.IMAGE
            ),
            num_hands=1,
            min_hand_detection_confidence=0.50,
            min_hand_presence_confidence=0.50
        )
    )

    return (
        mp.tasks.vision.HandLandmarker
        .create_from_options(options)
    )


hand_detector = load_hand_detector()


# ============================================================
# EXTRACT NORMALIZED HAND LANDMARKS
# ============================================================

def extract_landmarks(image):

    rgb_image = np.array(
        image.convert("RGB")
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    detection_result = (
        hand_detector.detect(mp_image)
    )

    if not detection_result.hand_landmarks:

        return None, None

    hand = (
        detection_result.hand_landmarks[0]
    )

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

    # Same normalization used during training
    max_value = np.max(
        np.abs(landmarks)
    )

    if max_value > 0:

        landmarks /= max_value

    return landmarks, hand


# ============================================================
# IR HAND DETECTION FALLBACK
# ============================================================

def detect_ir_hand(image):

    rgb_image = np.array(
        image.convert("RGB")
    )

    gray = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY
    )

    mean_intensity = np.mean(gray)

    bright_ratio = np.mean(
        gray > 180
    )

    infrared_like = (
        mean_intensity < 100
        and bright_ratio > 0.03
        and bright_ratio < 0.35
    )

    if not infrared_like:

        return False

    _, binary = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:

        return False

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    contour_area = cv2.contourArea(
        largest_contour
    )

    image_area = (
        gray.shape[0]
        * gray.shape[1]
    )

    area_ratio = (
        contour_area
        / image_area
    )

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    aspect_ratio = (
        w / float(h)
    )

    if (
        area_ratio > 0.03
        and area_ratio < 0.60
        and aspect_ratio > 0.20
        and aspect_ratio < 4.0
    ):

        return True

    return False


# ============================================================
# TITLE
# ============================================================

st.title(
    "✋ Hand Gesture Recognition"
)

st.write(
    "Upload a hand gesture image and "
    "the trained landmark model will "
    "predict the gesture."
)

st.info(
    "Hand Detection → 21 Landmarks → "
    "Random Forest Classification | "
    "8 Static Gestures + Motion Classes"
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a hand gesture image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    if st.button(
        "🔍 Predict Gesture",
        type="primary"
    ):

        # ----------------------------------------------------
        # STEP 1: MediaPipe hand detection
        # ----------------------------------------------------

        with st.spinner(
            "Checking for a hand..."
        ):

            landmarks, hand = (
                extract_landmarks(image)
            )

        # ----------------------------------------------------
        # NORMAL RGB HAND IMAGE
        # ----------------------------------------------------

        if landmarks is not None:

            st.success(
                "✋ Hand detected!"
            )

            # ------------------------------------------------
            # Landmark classification
            # ------------------------------------------------

            with st.spinner(
                "Analyzing hand landmarks..."
            ):

                input_data = (
                    landmarks.reshape(1, -1)
                )

                prediction = (
                    landmark_model.predict(
                        input_data
                    )[0]
                )

                probabilities = (
                    landmark_model.predict_proba(
                        input_data
                    )[0]
                )

            predicted_index = int(
                prediction
            )

            predicted_class = (
                landmark_class_names[
                    predicted_index
                ]
            )

            confidence = float(
                probabilities[
                    predicted_index
                ]
            )

            # ------------------------------------------------
            # Main result
            # ------------------------------------------------

            st.success(
                f"✋ Predicted Gesture: "
                f"**{predicted_class}**"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            # ------------------------------------------------
            # Top 3
            # ------------------------------------------------

            st.subheader(
                "Top 3 Predictions"
            )

            top_indices = np.argsort(
                probabilities
            )[-3:][::-1]

            for rank, idx in enumerate(
                top_indices,
                start=1
            ):

                gesture = (
                    landmark_class_names[idx]
                )

                score = float(
                    probabilities[idx]
                )

                st.write(
                    f"**{rank}. {gesture}** — "
                    f"{score * 100:.2f}%"
                )

                st.progress(
                    min(max(score, 0.0), 1.0)
                )

        # ----------------------------------------------------
        # NO MEDIAPIPE HAND
        # ----------------------------------------------------

        else:

            # Try LeapGestRecog IR fallback
            ir_hand = detect_ir_hand(
                image
            )

            if ir_hand and ir_model is not None:

                st.success(
                    "✋ IR-style hand detected!"
                )

                st.info(
                    "Using the original "
                    "LeapGestRecog MobileNetV2 "
                    "model for this IR-style image."
                )

                # --------------------------------------------
                # Prepare IR image
                # --------------------------------------------

                gray = image.convert("L")

                gray = gray.resize(
                    IR_IMG_SIZE
                )

                gray = np.array(
                    gray,
                    dtype=np.float32
                )

                gray = np.expand_dims(
                    gray,
                    axis=-1
                )

                gray = np.repeat(
                    gray,
                    3,
                    axis=-1
                )

                # MobileNetV2 preprocessing
                gray = (
                    gray / 127.5
                ) - 1.0

                gray = np.expand_dims(
                    gray,
                    axis=0
                )

                with st.spinner(
                    "Analyzing IR hand gesture..."
                ):

                    predictions = (
                        ir_model.predict(
                            gray,
                            verbose=0
                        )[0]
                    )

                predicted_index = int(
                    np.argmax(predictions)
                )

                predicted_class = (
                    ir_class_names[
                        predicted_index
                    ]
                )

                confidence = float(
                    predictions[
                        predicted_index
                    ]
                )

                st.success(
                    f"✋ Predicted Gesture: "
                    f"**{predicted_class}**"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            else:

                # --------------------------------------------
                # Not a hand
                # --------------------------------------------

                st.error(
                    "❌ This is not a hand."
                )

                st.warning(
                    "Please upload a clear hand "
                    "gesture image."
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "About the Project"
    )

    st.write(
        "Hand Gesture Recognition using "
        "MediaPipe hand landmarks and "
        "Random Forest classification."
    )

    st.write(
        "**Dataset:** LeapGestRecog"
    )

    st.write(
        "**Primary Model:** MobileNetV2"
    )

    st.write(
        "**Official Test Accuracy:** 84.55%"
    )

    st.write(
        "**Real-Time Model:** "
        "MediaPipe + Random Forest"
    )

    st.write(
        "**Landmark Test Accuracy:** 100%"
    )

    st.write(
        "**Static Gestures:** 8"
    )

    st.write(
        "**Final Gesture Classes:** 10"
    )

    st.write(
        "**Hand Detection:** MediaPipe"
    )

    st.divider()

    st.caption(
        "Developed as part of "
        "AI/ML Internship — Task 5"
    )