import cv2
import numpy as np
from ultralytics import YOLO
import torch

def detect_fall(frame, model, draw_on=None):
    """
    Detects if a person has fallen based on pose estimation.
    Returns: {"fall_detected": bool, "confidence": float, "boxes": list}
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = model(frame, imgsz=640, device=device, verbose=False)
    
    fall_detected = False
    confidence = 0.0
    detected_boxes = []

    scale_w = 1.0
    scale_h = 1.0
    if draw_on is not None:
        scale_w = draw_on.shape[1] / frame.shape[1]
        scale_h = draw_on.shape[0] / frame.shape[0]
        canvas = draw_on
    else:
        canvas = frame

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            conf = float(box.conf[0])

            # Check aspect ratio (width > height indicates horizontal posture)
            aspect_ratio = w / h if h > 0 else 0.0
            is_fall = aspect_ratio > 1.2 and conf > 0.5
            
            if is_fall:
                fall_detected = True
                confidence = max(confidence, conf)
                
            detected_boxes.append({
                "box": [x1, y1, x2, y2],
                "conf": conf,
                "is_fall": is_fall
            })

            # Draw on canvas using scaled coordinates
            sx1, sy1, sx2, sy2 = int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)
            if is_fall:
                cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
                cv2.putText(canvas, f"Fall Risk Detected! ({conf:.2f})", (sx1, sy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
                cv2.putText(canvas, f"Person ({conf:.2f})", (sx1, sy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return {
        "fall_detected": fall_detected,
        "confidence": confidence,
        "boxes": detected_boxes
    }

if __name__ == "__main__":
    # Test script for local run
    print("Loading YOLOv8 Pose model...")
    model = YOLO('yolov8n-pose.pt')
    
    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        res = detect_fall(frame, model)
        cv2.imshow("Fall Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
