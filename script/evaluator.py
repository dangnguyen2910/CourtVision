from ultralytics import YOLO 
from deep_sort_realtime.deepsort_tracker import DeepSort

import os
import cv2
import numpy as np
import pandas as pd
import json
import motmetrics as mm


class Evaluator: 
    def __init__(self): 
        self.model = YOLO("runs/detect/train3/weights/best.pt")
        self.tracker = DeepSort(max_age=30)
        self.TRACKER_METHOD = "botsort.yaml"     # "bytetrack.yaml" or "botsort.yaml"
        self.IoU_THRESHOLD = 0.76
        
        self.DETECTION_DATA_PATH = "datasets/player_tracking/data.yaml"
        self.YOLO_GT_FOLDER = "datasets/video/train/labels"  # Folder containing YOLO ground truth (.txt files)
        self.VIDEO_PATH = "datasets/video/sample.mp4"  
        self.MOT_GT_PATH = "datasets/video/mot_gt.csv"  # Output MOT format file
        
        self.DETECTION_RESULT_PATH = "outputs/detection_result.json"
        self.TRACK_RESULT_PATH = "outputs/track_result.csv"
        self.TRACK_METRIC_PATH = "outputs/track_metrics.csv"
        
        
    def run(self): 
        # self.yolo_to_mov()
        self.evaluate_detect()
        self.generate_tracking_result()
        self.evaluate_tracking()
        
        
    def yolo_to_mov(self): 
        # Read video frames
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Convert YOLO labels to MOT format
        mot_annotations = []
        gt_files_list = os.listdir(self.YOLO_GT_FOLDER)
        gt_files_list.sort()

        for frame_id in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            yolo_label_path = os.path.join(self.YOLO_GT_FOLDER, gt_files_list[frame_id])
            detections = []

            if os.path.exists(yolo_label_path):
                with open(yolo_label_path, "r") as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])

                        # Convert normalized YOLO format to absolute pixel coordinates
                        x1 = int((x_center - width / 2) * frame_width)
                        y1 = int((y_center - height / 2) * frame_height)
                        x2 = int((x_center + width / 2) * frame_width)
                        y2 = int((y_center + height / 2) * frame_height)
                        
                        w = x2 - x1
                        h = y2 - y1

                        detections.append(([x1, y1, w, h], 1.0, class_id))  # Confidence = 1.0 (since GT has no confidence)

            # Use DeepSORT to track objects and maintain consistent IDs
            tracks = self.tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                # if not track.is_confirmed():
                    # continue

                object_id = track.track_id
                x1, y1, w,h = map(int, track.to_ltwh())
                
                # w = x2 - x1
                # h = y2 - y1

                mot_annotations.append([frame_id, object_id, x1, y1, w, h, 1.0, class_id, -1, -1])

        cap.release()

        # Save to MOT format CSV
        df = pd.DataFrame(mot_annotations, columns=["frame", "id", "x1", "y1", "w", "h", "conf", "class", "-1", "-1"])
        df.to_csv(self.MOT_GT_PATH, index=False)

        
        
    def evaluate_detect(self): 
        """ 
        Calculate metrics for object detection
        """
        results = self.model.val(
            data = self.DETECTION_DATA_PATH, 
            split = "test"
        )

        box_metrics = {
            "mAP50": float(results.box.map50),       
            "mAP50-95": float(results.box.map),      
            "Precision": results.box.p.tolist(),  
            "Recall": results.box.r.tolist(),      
            "Per-Class mAP": results.box.maps.tolist() 
        }

        # Create outputs folder in case it's not already available. 
        if (not os.path.exists("outputs")): 
            os.makedirs("outputs")

        with open(self.DETECTION_RESULT_PATH, "w") as f: 
            json.dump(box_metrics, f, indent=4)
            
    
    def generate_tracking_result(self): 
        """ 
        Generate tracking result in MOT format
        """
        # Define a list to store tracking results
        tracking_results = []

        cap = cv2.VideoCapture(self.VIDEO_PATH)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(frame, persist=True, tracker=self.TRACKER_METHOD)

            # Save tracking results: [frame, track_id, x1, y1, x2, y2]
            if results[0].boxes.id is not None:
                for box, track_id in zip(results[0].boxes.xywh, results[0].boxes.id):
                    x1, y1, w, h = map(int, box.tolist())
                    track_id = int(track_id)
                    tracking_results.append([frame_idx, track_id, x1, y1, w, h])

            frame_idx += 1

        cap.release()

        # Convert results to DataFrame and save as CSV
        df_tracking = pd.DataFrame(
            tracking_results, 
            columns=["frame", "id", "x1", "y1", "w", "h"]
        )
        df_tracking.to_csv(self.TRACK_RESULT_PATH, index=False)
        
        
    def evaluate_tracking(self): 
        gt_df = pd.read_csv(self.MOT_GT_PATH)

        # Initialize MOTAccumulator
        acc = mm.MOTAccumulator(auto_id=True)

        # Load tracking results
        tracker_df = pd.read_csv(self.TRACK_RESULT_PATH)

        # Evaluate per frame
        for frame in sorted(gt_df["frame"].unique()):
            gt_frame = gt_df[gt_df["frame"] == frame]
            tracker_frame = tracker_df[tracker_df["frame"] == frame]

            # Get object IDs
            gt_ids = gt_frame["id"].values
            tracker_ids = tracker_frame["id"].values

            # Compute IoU distances
            dists = mm.distances.iou_matrix(
                gt_frame[["x1", "y1", "w", "h"]].values,
                tracker_frame[["x1", "y1", "w", "h"]].values,
                max_iou=self.IoU_THRESHOLD
            )

            # Update accumulator
            acc.update(gt_ids, tracker_ids, dists)

        # Compute tracking metrics
        mh = mm.metrics.create()
        summary = mh.compute(
            acc, 
            metrics=[
                "num_frames", 
                "num_matches",
                "num_switches", 
                "num_false_positives", 
                "num_misses",
                "mota", 
                "motp", 
                "idf1", 
                "precision", 
                "recall"
            ], 
            name="Tracking Evaluation"
        )

        res = pd.DataFrame.from_dict(summary).T
        print(res)
        res.to_csv(self.TRACK_METRIC_PATH)
        return res 
    


if __name__ == "__main__": 
    Evaluator().run()
