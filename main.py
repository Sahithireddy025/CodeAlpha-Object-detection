import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # We will create this file

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize SORT tracker
tracker = Sort()

# Open Webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    detections = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, score])

    detections = np.array(detections)

    # Update tracker
    tracked_objects = tracker.update(detections)

    # Draw tracking results
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj.astype(int)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID: {track_id}", 
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

    cv2.imshow("Object Detection & Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()