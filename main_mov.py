import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
from datetime import datetime


class brainiac:
    def __init__(self):
        # Modelo para detecção, Rastreador de objetos, Gerador de caixas e Rotulos.
        self.model = YOLO("yolov8n.pt")
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # Dicionario para guardar todas entidades detectadas e um contador para as mesmas.
        self.entity_detected = {}
        self.countdown = 30  # Frames

    # Metodo callback para o processamento dos quadros.
    def callback(self, frame: np.ndarray, _: int) -> np.ndarray:
        # Realiza as detecções, Converte os resultados, Filtra as detecções e depois atualiza  o rastreador.
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]
        detections = self.tracker.update_with_detections(detections)

        # Pega o horario atual
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        labels = []

        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            labels.append(f"#{tracker_id} {results.names[class_id]}")

            # detecção de entrada
            if not tracker_id in list(self.entity_detected.keys()):
                self.entity_detected[tracker_id] = self.countdown
                print(f"{tracker_id} entrou")
                with open("presence.txt", "a") as file:
                    file.write(f"\nPessoa {tracker_id} entrou - {current_time}")

        # detecção de saída
        for i in list(self.entity_detected.keys()):
            if not i in detections.tracker_id:
                self.entity_detected[i] -= 1
                if self.entity_detected[i] == 0:
                    del self.entity_detected[i]
                    print(f"{i} Saiu")
                    with open("presence.txt", "a") as file:
                        file.write(f"\nPessoa {i} saiu - {current_time}")

        annotated_frame = self.box_annotator.annotate(
            frame.copy(), detections=detections
        )
        return self.label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels
        )

    # Metodo para execução do video.
    def mov_exec(self):
        # Live Video
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process and annotate the frame
            annotated_frame = self.callback(frame, 0)

            # Display the resulting frame
            cv2.imshow("Frame", annotated_frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# Instances
if __name__ == "__main__":
    b = brainiac()
    b.mov_exec()
