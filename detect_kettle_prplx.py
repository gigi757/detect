import cv2
from ultralytics import YOLO
from collections import defaultdict

# Загрузка модели
model = YOLO('best.pt')  # Ваша обученная модель
track_history = defaultdict(lambda: [])
movement_status = defaultdict(lambda: False)
MOVEMENT_THRESHOLD = 5  # Порог смещения в пикселях
STOP_FRAMES = 10        # Кадров без движения для подтверждения остановки

# Инициализация видеопотока
cap = cv2.VideoCapture("vys40-5min.mp4")

frame_count = 0
stop_counter = defaultdict(int)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

 
    results = model.track(
        source=frame,
        persist=True,  # Для сохранения ID между кадрами
        conf=0.5,
        imgsz=640,
        verbose=False
    )

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            current_pos = (float(x), float(y))
            
            # Получаем историю позиций для этого трека
            track = track_history[track_id]
            track.append(current_pos)
            
            if len(track) > 2:  # Нужно минимум 2 точки для сравнения
                # Рассчитываем смещение от предыдущей позиции
                prev_pos = track[-2]
                dx = abs(current_pos[0] - prev_pos[0])
                dy = abs(current_pos[1] - prev_pos[1])
                displacement = (dx**2 + dy**2)**0.5
                
                # Определяем начало движения
                if not movement_status[track_id] and displacement > MOVEMENT_THRESHOLD:
                    print(f"Frame {frame_count}: Движение началось (ID {track_id})")
                    movement_status[track_id] = True
                    stop_counter[track_id] = 0
                
                # Определяем окончание движения
                if movement_status[track_id]:
                    if displacement < MOVEMENT_THRESHOLD:
                        stop_counter[track_id] += 1
                        if stop_counter[track_id] >= STOP_FRAMES:
                            print(f"Frame {frame_count}: Движение прекращено (ID {track_id})")
                            movement_status[track_id] = False
                            stop_counter[track_id] = 0
                    else:
                        stop_counter[track_id] = 0

                # Визуализация
                status = "MOVING" if movement_status[track_id] else "STOPPED"
                color = (0, 255, 0) if status == "MOVING" else (0, 0, 255)
                cv2.putText(annotated_frame, f"ID {track_id}: {status}", 
                           (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)

        cv2.imshow('Kettlebell Tracking', annotated_frame)
    else:
        cv2.imshow('Kettlebell Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
