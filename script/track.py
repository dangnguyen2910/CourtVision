from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")

results = model.track(
    "datasets/video/sample.mp4", 
    show=True, 
    save=True
)
