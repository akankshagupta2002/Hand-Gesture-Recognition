import cv2
import numpy as np
import os

input_file = "results/demo/hand_gesture_demo.mp4"
output_file = "results/demo/hand_gesture_demo_final.mp4"

cap = cv2.VideoCapture(input_file)

if not cap.isOpened():
    print("ERROR: Could not open original video.")
    raise SystemExit

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = cap.get(cv2.CAP_PROP_FPS)

# 2x slower
output_fps = original_fps / 2

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_file,
    fourcc,
    output_fps,
    (width, height)
)

if not writer.isOpened():
    print("ERROR: Could not create output video.")
    cap.release()
    raise SystemExit

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # 1. Slight brightness + contrast improvement
    # --------------------------------------------------------

    # Very mild adjustment to keep the video natural
    enhanced = cv2.convertScaleAbs(
        frame,
        alpha=1.05,
        beta=5
    )

    # --------------------------------------------------------
    # 2. Mild sharpening
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.20,
        blurred,
        -0.20,
        0
    )

    # --------------------------------------------------------
    # 3. Save frame
    # --------------------------------------------------------

    writer.write(sharpened)

cap.release()
writer.release()

print()
print("=" * 55)
print("FINAL DEMO VIDEO CREATED")
print("=" * 55)
print()
print("Slow: 2x")
print("Brightness: Mild")
print("Contrast: Mild")
print("Sharpening: Mild")
print()
print("Saved at:")
print(output_file)
print()