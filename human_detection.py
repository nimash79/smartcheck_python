from ultralytics import YOLO
import cv2

# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")  # 'n' = nano version (fast), you can use yolov8s.pt, yolov8m.pt, etc.

# Open webcam (use 0) or video file path
cap = cv2.VideoCapture("rtsp://admin:@192.168.0.197:554/live/ch00_0")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Filter detections for "person" class
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label == "person":  # only keep humans
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
