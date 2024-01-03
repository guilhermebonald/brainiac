import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
# trace_annotator = sv.TraceAnnotator()

entity_detected = []


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = []

    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
        labels.append(f"#{tracker_id} {results.names[class_id]}")

        # detecção de entrada
        if not tracker_id in entity_detected:
            entity_detected.append(tracker_id)
            print(f"{tracker_id} entrou")

    # detecção de saída
    for i in entity_detected:
        if not i in detections.tracker_id:
            entity_detected.remove(i)
            print(f"{i} saiu")

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels
    )


# Video File
sv.process_video(
    source_path="./media/cars2.mp4", target_path="result.mp4", callback=callback
)


# # Live Video
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Process and annotate the frame
#     annotated_frame = callback(frame, 0)

#     # Display the resulting frame
#     cv2.imshow("Frame", annotated_frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
