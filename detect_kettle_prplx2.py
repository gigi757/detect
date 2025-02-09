import cv2
from ultralytics import YOLO
from collections import defaultdict

# Загрузка обученной модели
model = YOLO('best.pt') 

# Параметры детекции
MOVEMENT_THRESHOLD = 3  # Порог смещения в пикселях
STOP_FRAMES = 3         # Кадров без движения для подтверждения остановки

# Инициализация трекера
track_history = defaultdict(lambda: {
    'positions': [],
    'moving': False,
    'stop_counter': 0
})

cap = cv2.VideoCapture('video_2025-02-01_12-10-52.mp4') 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция и трекинг
    results = model.track(
        frame,
        persist=True,
        conf=0.5,
        classes=[0],
        imgsz=640,
        verbose=False
    )

    annotated_frame = frame.copy()
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            current_pos = (float(x), float(y))
            
            # Обновляем историю позиций
            history = track_history[track_id]
            history['positions'].append(current_pos)
            
            # Проверка движения
            if len(history['positions']) > 1:
                prev_pos = history['positions'][-2]
                dx = abs(current_pos[0] - prev_pos[0])
                dy = abs(current_pos[1] - prev_pos[1])
                displacement = (dx**2 + dy**2)**0.5

                if displacement > MOVEMENT_THRESHOLD:
                    if not history['moving']:
                        print("Гиря начала движение")
                    history['moving'] = True
                    history['stop_counter'] = 0
                else:
                    history['stop_counter'] += 1
                    if history['moving'] and history['stop_counter'] >= STOP_FRAMES:
                        print("Гиря закончила движение")
                        history['moving'] = False

            # Отрисовка статуса
            status = "MOVING" if history['moving'] else "STOPPED"
            color = (0, 255, 0) if status == "MOVING" else (0, 0, 255)
            cv2.rectangle(annotated_frame, 
                        (int(x - w/2), int(y - h/2)),
                        (int(x + w/2), int(y + h/2)),
                        color, 2)

    cv2.imshow('Kettlebell Tracking', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
