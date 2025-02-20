import cv2
import numpy as np

# Load ground truth file
gt_file = "datasets/video/mot_gt.csv"  # Path to MOT ground truth file
video_path = "datasets/video/sample.mp4"  # Path to input video
output_path = "datasets/video/sample_annotated.mp4"  # Path to save output

# Read ground truth data into a dictionary
gt_data = {}
line_id = 0
with open(gt_file, "r") as f:
    for line in f:
        if (line_id == 0):
            line_id += 1
            continue 
        
        values = line.strip().split(",")
        frame_id = int(values[0])
        object_id = int(values[1])
        x, y, w, h = map(int, map(float, values[2:6]))  # Convert to integers

        if frame_id not in gt_data:
            gt_data[frame_id] = []
        gt_data[frame_id].append((object_id, x, y, w, h))

# Open video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_id = 1  # MOT format starts from 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Draw ground truth bounding boxes if present in current frame
    if frame_id in gt_data:
        for obj in gt_data[frame_id]:
            object_id, x, y, w, h = obj
            color = (0, 255, 0)  # Green color for GT boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(frame, f"ID: {object_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Write frame to output video
    out.write(frame)

    # Show frame (optional)
    cv2.imshow("Ground Truth", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1  # Move to the next frame

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
