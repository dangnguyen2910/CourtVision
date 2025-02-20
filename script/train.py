from ultralytics import YOLO


if __name__ == "__main__": 
    model = YOLO("yolo11n.pt")
    model.train(
        data = "datasets/player_tracking/data.yaml", 
        epochs=2, 
        imgsz=640, 
        batch=1,
        cos_lr = True, 
        lr0 = 0.001, 
        lrf = 0.0001, 
        patience = 10
    )

