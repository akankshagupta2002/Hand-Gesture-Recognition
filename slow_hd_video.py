import cv2
import os

input_file = "results/demo/hand_gesture_demo_HD.mp4"
output_file = "results/demo/hand_gesture_demo_HD_slow.mp4"

# 1.25x slower than current HD video
SLOW_FACTOR = 1.25

cap = cv2.VideoCapture(input_file)

if not cap.isOpened():
    print("ERROR: Could not open HD video.")
    raise SystemExit

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = cap.get(cv2.CAP_PROP_FPS)

if original_fps <= 0:
    original_fps = 15.0

output_fps = original_fps / SLOW_FACTOR

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

    writer.write(frame)

cap.release()
writer.release()

print()
print("=" * 55)
print("SLOW HD DEMO CREATED")
print("=" * 55)
print()
print("Additional slow-down: 1.25x")
print("Resolution:", width, "x", height)
print()
print("Saved at:")
print(output_file)