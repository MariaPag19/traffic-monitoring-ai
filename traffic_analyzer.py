from ultralytics import YOLO
import cv2

# carica il modello YOLO
model = YOLO("yolov8n.pt")

# apri il video
cap = cv2.VideoCapture("traffic.mp4")

# posizione della linea di conteggio
line_y = 430

# contatore veicoli
vehicle_count = 0

# memorizza posizione precedente dei veicoli
previous_positions = {}

# crea finestra ridimensionabile
cv2.namedWindow("Traffic Monitoring System", cv2.WINDOW_NORMAL)

while True:

    # legge un frame del video
    ret, frame = cap.read()

    if not ret:
        break

    # ridimensiona il frame per evitare zoom
    frame = cv2.resize(frame, (960, 540))

    # detection + tracking
    results = model.track(frame, persist=True)

    boxes = results[0].boxes

    if boxes.id is not None:

        ids = boxes.id.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        for box, id in zip(xyxy, ids):

            x1, y1, x2, y2 = box

            # centro della bounding box
            center_y = int((y1 + y2) / 2)

            # controlla se il veicolo era già stato visto
            if id in previous_positions:

                prev_y = previous_positions[id]

                # controlla se attraversa la linea
                if prev_y < line_y and center_y >= line_y:

                    vehicle_count += 1

            # aggiorna posizione precedente
            previous_positions[id] = center_y

    # disegna bounding boxes
    annotated = results[0].plot()

    # disegna linea di conteggio
    cv2.line(annotated, (0, line_y), (960, line_y), (0, 0, 255), 3)

    # mostra il numero di veicoli
    cv2.putText(
        annotated,
        f"Vehicles: {vehicle_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # mostra il video
    cv2.imshow("Traffic Monitoring System", annotated)

    # premi Q per uscire
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# chiude video e finestre
cap.release()
cv2.destroyAllWindows()