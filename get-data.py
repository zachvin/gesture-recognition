import mediapipe as mp
import cv2
import time
import pandas as pd
import numpy as np

model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

current_gesture = -1

# Create a hand landmarker instance with the live stream mode:
def record_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # make sure a gesture is being recorded
    if current_gesture not in ['thumbs up', 'thumbs down', 'stop', 'excuse me']:
       return
    
    # save all landmarks in res
    if result.hand_landmarks:
        res_x = np.zeros(21)
        res_y = np.zeros(21)
        for i,lmk in enumerate(result.hand_landmarks[0]):
            res_x[i] = lmk.x
            res_y[i] = lmk.y
    else:
       return

    # normalize landmarks
    res_x = (res_x - res_x.min()) / (res_x.max() - res_x.min())
    res_y = (res_y - res_y.min()) / (res_y.max() - res_y.min())
    res = []
    for x,y in zip(res_x, res_y):
       res += [x, y]
    res.append(current_gesture)
    data.loc[len(data)] = res

    print(f'Gesture {current_gesture} recorded.')
    

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=record_result)

# Data collection
cols = []
for lmk in range(1,22):
   for dir in ['x','y']:
      cols.append(f'{lmk}{dir}')
cols.append('gesture')

data = pd.DataFrame(columns=cols)

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

      if k == ord('a'):
         current_gesture = 'thumbs up'
      elif k == ord('b'):
         current_gesture = 'thumbs down'
      elif k == ord('c'):
         current_gesture = 'stop'
      elif k == ord('d'):
         current_gesture = 'excuse me'
      else:
         current_gesture = -1
       
if len(data) > 0:
   data.to_csv('data/gesture-data.csv', mode='a', header=False, index=None)
cv2.destroyAllWindows()
cap.release()