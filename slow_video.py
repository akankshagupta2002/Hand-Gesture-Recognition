import cv2

input_file = "results/demo/hand_gesture_demo.mp4"
output_file = "results/demo/hand_gesture_demo_slow.mp4"

cap = cv2.VideoCapture(input_file)

if not cap.isOpened():
    print("ERROR: Could not open original video.")
    raise SystemExit

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Original video FPS
original_fps = cap.get(cv2.CAP_PROP_FPS)

# 2x slower playback
slow_fps = original_fps / 2

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_file,
    fourcc,
    slow_fps,
    (width, height)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    writer.write(frame)

cap.release()
writer.release()

print()
print("Slow video created successfully!")
print()
print("Saved at:")
print(output_file)