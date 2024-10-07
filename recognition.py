import mediapipe as mp
import cv2
import time
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pickle

model_path = 'models/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

classifier = None
with open('models/classifier.pkl', 'rb') as f:
   classifier = pickle.load(f)

prediction_classifier = {
   0: 'stop',
   1: 'thumbs down',
   2: 'thumbs up',
   3: 'excuse me',
}

cols = []
for lmk in range(1,22):
   for dir in ['x','y']:
      cols.append(f'{lmk}{dir}')

# Create a hand landmarker instance with the live stream mode:
def classify_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):    
   # save all landmarks in res
   print(len(result.hand_landmarks))
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
   res = [res]

   res = pd.DataFrame(res, columns=cols)
   pred = classifier.predict(res)
   print(f'Detected {prediction_classifier.get(pred[0], -1)}')

    

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=classify_result)

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