import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

# Module-level state for temporal analysis
_last_call_time = 0.0
_fire_area_history = []  # List of total active fire pixel areas in the last 5 frames

def detect_fire(frame, model, draw_on=None):
    """
    Detects fire or smoke. 
    Returns: {"fire_detected": bool, "confidence": float, "boxes": list, "contours": list}
    """
    global _last_call_time, _fire_area_history
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = model(frame, imgsz=384, device=device, verbose=False)
    
    fire_detected = False
    confidence = 0.0
    detected_boxes = []
    detected_contours = []
    person_boxes = []

    scale_w = 1.0
    scale_h = 1.0
    if draw_on is not None:
        scale_w = draw_on.shape[1] / frame.shape[1]
        scale_h = draw_on.shape[0] / frame.shape[0]
        canvas = draw_on
    else:
        canvas = frame

    # AI Detection Logic
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            if label in ['fire', 'smoke']:
                if conf >= 0.65:
                    fire_detected = True
                    confidence = max(confidence, conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detected_boxes.append({
                        "box": [x1, y1, x2, y2],
                        "conf": conf,
                        "label": label
                    })
                    sx1, sy1, sx2, sy2 = int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)
                    cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (0, 165, 255), 2)
                    cv2.putText(canvas, f"{label.capitalize()} ({conf:.2f})", (sx1, sy1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            elif label == 'person' and conf > 0.4:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                person_boxes.append((x1, y1, x2, y2))

    # Simple Color-based fallback
    if not fire_detected:
        current_time = time.time()
        if current_time - _last_call_time > 1.5:
            _fire_area_history.clear()
        _last_call_time = current_time

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Balanced HSV Range: Hue 10-30, Saturation 110-255, Value 150-255
        lower_fire = np.array([10, 110, 150], dtype="uint8")
        upper_fire = np.array([30, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, lower_fire, upper_fire)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Adaptive area threshold: 0.00008 of total image area, min 20 pixels
        img_area = frame.shape[0] * frame.shape[1]
        min_area = max(20.0, float(img_area) * 0.00008)

        valid_fire_area = 0
        fire_contours_found = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0.0

                if solidity < 0.88:
                    perimeter = cv2.arcLength(contour, True)
                    complexity = (perimeter ** 2) / area if area > 0 else 0.0
                    
                    if complexity > 45:
                        x, y, w, h = cv2.boundingRect(contour)
                        on_person = False
                        for (px1, py1, px2, py2) in person_boxes:
                            ix1 = max(x, px1)
                            iy1 = max(y, py1)
                            ix2 = min(x + w, px2)
                            iy2 = min(y + h, py2)
                            if ix2 > ix1 and iy2 > iy1:
                                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                                if intersection_area / (w * h) > 0.20:
                                    on_person = True
                                    break
                                    
                        if not on_person:
                            valid_fire_area += area
                            fire_contours_found.append(contour)

        _fire_area_history.append(valid_fire_area)
        if len(_fire_area_history) > 5:
            _fire_area_history.pop(0)

        # Temporal Flicker Check
        if len(_fire_area_history) >= 4 and all(a > 0 for a in _fire_area_history):
            mean_area = np.mean(_fire_area_history)
            std_area = np.std(_fire_area_history)
            cv = std_area / mean_area if mean_area > 0 else 0.0

            if cv > 0.04:
                fire_detected = True
                confidence = 0.92

                for contour in fire_contours_found:
                    detected_contours.append({
                        "contour": contour.tolist(),
                        "flicker": cv
                    })
                    
                    scale_array = np.array([scale_w, scale_h])
                    scaled_contour = (contour * scale_array).astype(np.int32)
                    x, y, w, h = cv2.boundingRect(scaled_contour)
                    cv2.drawContours(canvas, [scaled_contour], -1, (0, 165, 255), 2)
                    cv2.putText(canvas, f"Flame (Flicker: {cv:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    return {
        "fire_detected": fire_detected,
        "confidence": confidence,
        "boxes": detected_boxes,
        "contours": detected_contours
    }

if __name__ == "__main__":
    # Test script
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt') 
    
    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        res = detect_fire(frame, model)
        is_alert = res["fire_detected"]
        conf = res["confidence"]
        
        cv2.imshow("Fire Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
