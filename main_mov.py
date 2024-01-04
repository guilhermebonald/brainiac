import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
# trace_annotator = sv.TraceAnnotator()

entity_detected = {}
countdown = 30  # Frames


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]
    detections = tracker.update_with_detections(detections)

    labels = []

    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
        labels.append(f"#{tracker_id} {results.names[class_id]}")

        # detecção de entrada
        if not tracker_id in list(entity_detected.keys()):
            entity_detected[tracker_id] = countdown
            print(f"{tracker_id} entrou")
            with open("presence.txt", "a") as file:
                file.write(f"\nPessoa {tracker_id} entrou.")

    # detecção de saída
    for i in list(entity_detected.keys()):
        if not i in detections.tracker_id:
            entity_detected[i] -= 1
            if entity_detected[i] == 0:
                del entity_detected[i]
                print(f"{i} Saiu")
                with open("presence.txt", "a") as file:
                    file.write(f"\nPessoa {i} saiu.")

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels
    )


# # Video File
# sv.process_video(
#     source_path="./media/cars2.mp4", target_path="result.mp4", callback=callback
# )


# Live Video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process and annotate the frame
    annotated_frame = callback(frame, 0)

    # Display the resulting frame
    cv2.imshow("Frame", annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
