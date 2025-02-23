import time
from ultralytics import YOLO
import numpy as np 
import cv2 

# Load YOLO model
model = YOLO("runs/detect/train3/weights/best.pt")  # Replace with your model

# Run tracking on a video file
video_path = "datasets/video/sample.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))  # Width
frame_height = int(cap.get(4))  # Height
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if FPS is 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (use 'MP4V' for .mp4)
out = cv2.VideoWriter('outputs/output_cpu.avi', fourcc, fps, (frame_width, frame_height))

frame_count = 0
total_time = 0
fps_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()  # Start time

    results = model.track(frame, persist=True, device=0)  # YOLO tracking

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time
    fps = 1 / elapsed_time

    total_time += elapsed_time
    frame_count += 1

    print(f"Frame {frame_count} | FPS: {fps:.2f}")
    fps_list.append(fps)
    cv2.putText(frame, f"FPS: {fps:.2f}", (0, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            track_id = int(box.id[0]) if box.id is not None else -1  

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw Track ID
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    
    out.write(frame)
    cv2.imshow("frame", frame)
    
    if (cv2.waitKey(1) & 0xFF == ord('q')): 
        break
        
    

print(f"Mean fps: {np.mean(fps_list):.2f}")
out.release()
cap.release()
cv2.destroyAllWindows()
