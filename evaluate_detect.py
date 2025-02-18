from ultralytics import YOLO 
import json
import os


if __name__ == "__main__": 
    model = YOLO("runs/detect/train/train2/weights/best.pt")

    results = model.val(
        data = "datasets/data.yaml", 
        split = "test"
    )
    
    box_metrics = {
        "mAP50": float(results.box.map50),       # Convert to float
        "mAP50-95": float(results.box.map),      # Convert to float
        "Precision": results.box.p.tolist(),  # Convert to float
        "Recall": results.box.r.tolist(),      # Convert to float
        "Per-Class mAP": results.box.maps.tolist()  # Convert NumPy array to list
    }
    
    # Create outputs folder in case it's not already available. 
    if (not os.path.exists("outputs")): 
        os.makedirs("outputs")

    with open("outputs/detection_results_3.json", "w") as f: 
        json.dump(box_metrics, f, indent=4)
