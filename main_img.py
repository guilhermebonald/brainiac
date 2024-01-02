import cv2
from ultralytics import YOLO
import supervision as sv

# Modelo treinado.
model = YOLO("yolov8n.pt")
image = cv2.imread("./media/image2.jpg")

# Detecta objetos na imagem
results = model(image)[0]
# Convertendo os resultados da detecção de objetos do formato Ultralytics para o formato Supervision
detections = sv.Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Obtendo os nomes das classes dos objetos detectados
labels = [results.names[class_id] for class_id in detections.class_id]

# Anotando a imagem com os rótulos dos objetos detectados
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels
)

cv2.imwrite("image.png", annotated_image)
