from ultralytics import YOLO 

if __name__ == "__main__": 
    model = YOLO("runs/detect/train/train2/weights/best.pt")
    model.track(
        "video.mp4",
        save=True, 
        name="track_result2"
)
