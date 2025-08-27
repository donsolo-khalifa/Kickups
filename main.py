import cv2
import numpy as pd
from cvzone.PoseModule import PoseDetector
from ultralytics import YOLO
import cvzone

from cvzone.PlotModule import LivePlot

VIDEO_PATH = 'vidb.mp4'
YOLO_WEIGHTS = '../QuadraticParabola/yolo-weights/yolo11x.pt'
BALL_CLASS_ID = 32  # COCO index for sports ball

# --- Contact Detection Parameters ---
# The distance in pixels between ball and body part to trigger kick
CONTACT_THRESHOLD = 70
# Frames to wait after a kick is detected before counting another one
COOLDOWN_FRAMES = 13

# --- Initialization ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Could not open video {VIDEO_PATH}")

cv2.namedWindow("Kick-up Counter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Kick-up Counter", 900, 600)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector = PoseDetector()
model = YOLO(YOLO_WEIGHTS)

kickup_count = 0
cooldown_counter = 0

# 25, 26 = Knees | 27, 28 = Ankles | 31, 32 = Heels
CONTACT_LANDMARKS = [31, 32]

plotY = LivePlot(w=800, h=400, yLimit=[0, 200], invert=True, char='D')  # D for Distance

while True:
    success, img = cap.read()
    if not success:
        print(f"Video finished. Total Kick-ups: {kickup_count}")
        break
    # img = cv2.resize(img, (900,900))
    img = detector.findPose(img, draw=False)
    lmList, _ = detector.findPosition(img, draw=False)

    results = model(img, stream=False, verbose=False)
    min_distance = float('inf')  # Reset minimum distance for this frame

    if cooldown_counter > 0:
        cooldown_counter -= 1

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == BALL_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
                ball_pos = (cx, cy)
                cv2.circle(img, ball_pos, 10, (0, 255, 0), cv2.FILLED)

                if lmList:
                    for lm_index in CONTACT_LANDMARKS:
                        lm_pos = lmList[lm_index][:2]
                        distance, _, _ = detector.findDistance(ball_pos, lm_pos, img, color=(255, 255, 0), scale=11)
                        min_distance = min(min_distance, distance)

                        if distance < CONTACT_THRESHOLD and cooldown_counter == 0:
                            kickup_count += 1
                            cooldown_counter = COOLDOWN_FRAMES
                            break
                break
        if min_distance != float('inf'):
            break

    # --- Visualization ---
    # Update the plot with the minimum distance value for this frame
    plot_color = (0, 200, 0) if cooldown_counter > 1 else (0, 0, 255)
    imgPlot = plotY.update(min_distance if min_distance != float('inf') else 200, color=plot_color)

    cvzone.putTextRect(img, f'Kick-ups: {kickup_count}', (50, 80), scale=3, thickness=3, colorR=(255, 255, 0),
                       colorT=(0, 0, 0), offset=20)

    # Resize and stack images
    img_resized = cv2.resize(img, (imgPlot.shape[1], imgPlot.shape[0]))
    imgStack = cvzone.stackImages([img_resized, imgPlot], 2, 1)

    cv2.imshow("Kick-up Counter", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
