import os
import json
import cv2
import joblib
import numpy as np
import streamlit as st
import mediapipe as mp
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PATHS
# ============================================================

LANDMARK_MODEL_PATH = os.path.join(
    "models", "hand_gesture_landmark_rf.joblib"
)

LANDMARK_CLASSES_PATH = os.path.join(
    "models", "landmark_class_names.json"
)

MOBILENET_MODEL_PATH = os.path.join(
    "models", "hand_gesture_mobilenetv2_improved.keras"
)

MOBILENET_CLASSES_PATH = os.path.join(
    "models", "class_names.json"
)

HAND_LANDMARKER_PATH = "hand_landmarker.task"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="✋",
    layout="wide"
)


# ============================================================
# LOAD RANDOM FOREST LANDMARK MODEL
# ============================================================

@st.cache_resource
def load_landmark_model():
    return joblib.load(LANDMARK_MODEL_PATH)


@st.cache_data
def load_landmark_classes():
    with open(LANDMARK_CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


landmark_model = load_landmark_model()
landmark_class_names = load_landmark_classes()


# ============================================================
# LOAD MOBILE NET V2 MODEL
# Used as fallback for LeapGestRecog IR-style images
# ============================================================

@st.cache_resource
def load_mobilenet_model():
    return tf.keras.models.load_model(MOBILENET_MODEL_PATH)


@st.cache_data
def load_mobilenet_classes():
    with open(MOBILENET_CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


mobilenet_model = load_mobilenet_model()
mobilenet_class_names = load_mobilenet_classes()


# ============================================================
# MEDIA PIPE HAND LANDMARKER
# ============================================================

@st.cache_resource
def load_hand_landmarker():

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=HAND_LANDMARKER_PATH
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )


hand_landmarker = load_hand_landmarker()


# ============================================================
# LANDMARK NORMALIZATION
# Same normalization used during Random Forest training
# ============================================================

def normalize_landmarks(hand_landmarks):

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32
    )

    # Wrist as origin
    wrist = points[0].copy()
    points = points - wrist

    # Scale normalization
    max_value = np.max(np.abs(points))

    if max_value > 0:
        points = points / max_value

    return points.flatten().reshape(1, -1)


# ============================================================
# MEDIA PIPE LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(image_rgb):

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = hand_landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]

    features = normalize_landmarks(hand)

    return features


# ============================================================
# DETECT IR-STYLE LEAPGESTRECOG IMAGE
# ============================================================

def is_ir_style_image(image_rgb):

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    mean_value = float(np.mean(gray))
    bright_ratio = float(
        np.mean(gray > 180)
    )

    # LeapGestRecog images generally have
    # bright hand region on dark background.
    return (
        mean_value < 120
        and bright_ratio > 0.08
    )


# ============================================================
# PREPARE IMAGE FOR MOBILE NET V2
# ============================================================

def prepare_mobilenet_image(image_rgb):

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    resized = cv2.resize(
        gray,
        (160, 160)
    )

    # Convert grayscale to 3 channels
    rgb_like = np.stack(
        [resized, resized, resized],
        axis=-1
    )

    rgb_like = rgb_like.astype(np.float32)

    rgb_like = preprocess_input(
        rgb_like
    )

    return np.expand_dims(
        rgb_like,
        axis=0
    )


# ============================================================
# MOBILE NET PREDICTION
# ============================================================

def predict_with_mobilenet(image_rgb):

    input_image = prepare_mobilenet_image(
        image_rgb
    )

    probabilities = mobilenet_model.predict(
        input_image,
        verbose=0
    )[0]

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    predictions = []

    for index in top_indices:

        predictions.append(
            (
                mobilenet_class_names[index],
                float(probabilities[index])
            )
        )

    return predictions


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

def predict_with_landmarks(features):

    probabilities = landmark_model.predict_proba(
        features
    )[0]

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    predictions = []

    for index in top_indices:

        predictions.append(
            (
                landmark_class_names[index],
                float(probabilities[index])
            )
        )

    return predictions


# ============================================================
# DISPLAY PREDICTIONS
# ============================================================

def display_predictions(predictions):

    if not predictions:
        return

    gesture = predictions[0][0]
    confidence = predictions[0][1]

    st.success(
        f"✋ Hand detected!\n\n"
        f"### {gesture}"
    )

    st.metric(
        "Model Probability",
        f"{confidence * 100:.2f}%"
    )

    st.markdown("### Top 3 Predictions")

    for name, score in predictions:

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write(name)

        with col2:
            st.write(
                f"{score * 100:.2f}%"
            )

        st.progress(
            min(max(score, 0.0), 1.0)
        )


# ============================================================
# HEADER
# ============================================================

st.title("✋ Hand Gesture Recognition")

st.write(
    "Upload a hand image to recognize the gesture "
    "using MediaPipe hand landmarks and machine learning."
)

st.info(
    "For normal RGB hand images, the application uses "
    "MediaPipe 21 hand landmarks + Random Forest. "
    "LeapGestRecog IR-style images use the MobileNetV2 "
    "evaluation model as a fallback."
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_rgb = np.array(image)

    st.markdown("### Uploaded Image")

    st.image(
        image,
        width=500
    )

    st.markdown("---")

    # --------------------------------------------------------
    # FIRST: IR DATASET IMAGE
    # --------------------------------------------------------

    if is_ir_style_image(image_rgb):

        st.info(
            "LeapGestRecog-style image detected. "
            "Using MobileNetV2 image classifier."
        )

        predictions = predict_with_mobilenet(
            image_rgb
        )

        display_predictions(
            predictions
        )

        st.caption(
            "Model: MobileNetV2 | "
            "Primary Task 5 evaluation model"
        )

    # --------------------------------------------------------
    # SECOND: NORMAL RGB IMAGE
    # --------------------------------------------------------

    else:

        features = extract_landmarks(
            image_rgb
        )

        # ----------------------------------------------------
        # HAND NOT FOUND
        # ----------------------------------------------------

        if features is None:

            st.error(
                "❌ This is not a hand"
            )

            st.warning(
                "Please upload an image containing "
                "a clearly visible hand."
            )

        # ----------------------------------------------------
        # HAND FOUND
        # ----------------------------------------------------

        else:

            st.success(
                "✋ Hand detected!"
            )

            predictions = predict_with_landmarks(
                features
            )

            display_predictions(
                predictions
            )

            st.caption(
                "Model: MediaPipe 21 Landmarks + "
                "Random Forest"
            )

            st.info(
                "Note: A single uploaded image represents "
                "a static frame. Moving gestures such as "
                "Palm Moved and Fist Moved are distinguished "
                "using temporal motion detection in the "
                "real-time webcam application."
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Project Information")

    st.markdown(
        """
        **Dataset:** LeapGestRecog

        **Task:** AI/ML Internship — Task 5

        **Gesture Classes:** 10

        **Primary Evaluation Model:** MobileNetV2

        **Official Test Accuracy:** 84.55%

        **Webcam Model:** MediaPipe Landmarks + Random Forest

        **Real-Time Detection:** Supported

        **Motion Detection:** Supported

        **Prediction Smoothing:** Supported
        """
    )

    st.markdown("---")

    st.subheader("Gesture Classes")

    for gesture in mobilenet_class_names:
        st.write(f"• {gesture}")

    st.markdown("---")

    st.caption(
        "Hand Gesture Recognition — Task 5"
    )