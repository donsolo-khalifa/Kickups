import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
from ultralytics import YOLO
import cvzone
from scipy.signal import find_peaks


VIDEO_PATH = '../PoseClassificationYolo/coolfreestyle.mp4' # ../QuadraticParabola/Videos/shortfreestyle.mp4  vidb.mp4  ../PoseClassificationYolo/coolfreestyle.mp4
YOLO_WEIGHTS = '../QuadraticParabola/yolo-weights/yolo11x.pt' #change path to yolo11x.pt
BALL_CLASS_ID = 32  # COCO index for sports ball
DISPLAY_WIDTH, DISPLAY_HEIGHT = 800, 600
SMOOTH_WINDOW = 7      # frames for moving average
PEAK_DISTANCE = 10     # minimum frames between kick-ups
PEAK_HEIGHT_FRAC = 0.1 # fraction of max height to count


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Could not open video {VIDEO_PATH}")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector = PoseDetector()
model = YOLO(YOLO_WEIGHTS)

# Pre-alloc arrays
ball_y = np.full((total_frames,), np.nan)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", DISPLAY_WIDTH, DISPLAY_HEIGHT)

frame_idx = 0
while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=False, verbose=False)
    detected = False
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == BALL_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = x1 + (x2 - x1) // 2
                cy = y1 + (y2 - y1) // 2
                ball_y[frame_idx] = cy
                detected = True

                cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), colorR=(255,255,0))
                cvzone.putTextRect(img, f"ball", (x1, y1-10), colorR=(255,255,0), scale=1)
                break
        if detected:
            break

    img = detector.findPose(img)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# Interpolate missing y-values
frames = np.arange(total_frames)
valid = ~np.isnan(ball_y)
if valid.sum() > 0:
    interp_y = np.interp(frames, frames[valid], ball_y[valid])
else:
    interp_y = np.zeros_like(ball_y)

# Convert to height signal
height = frame_height - interp_y

# Smooth signal
kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
height_smooth = np.convolve(height, kernel, mode='same')

# Peakaboo i see you detection
peaks, props = find_peaks(
    height_smooth,
    distance=PEAK_DISTANCE,
    height=PEAK_HEIGHT_FRAC * np.nanmax(height_smooth)
)
num_kickups = len(peaks)
print(f"Detected {num_kickups} kick-ups.")

# Plot for verification
try:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(height, alpha=0.3, label='raw height')
    plt.plot(height_smooth, label='smoothed')
    plt.plot(peaks, height_smooth[peaks], 'x', label='kick-ups')
    plt.legend()
    plt.xlabel('Frame index')
    plt.ylabel('Height (px)')
    plt.title('Kick-up detection')
    plt.show()
except ImportError:
    pass
