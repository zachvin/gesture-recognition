import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if result.hand_landmarks:
       print(result.hand_landmarks[0][0].x)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with HandLandmarker.create_from_options(options) as landmarker:
  
  base_time = time.time() * 1000.0
  cap = cv2.VideoCapture(0)
  while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imshow('frame', frame)

        timestamp = int((time.time() * 1000.0) - base_time)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        landmarker.detect_async(mp_image, timestamp)

    k = cv2.waitKey(1)
    if k == 27:
       break

cv2.destroyAllWindows()
cap.release()