import mediapipe as mp
import cv2
import time
import pandas as pd
import numpy as np

hand_model_path = 'models/hand_landmarker.task'
pose_model_path = 'models/pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode    

hand_options = HandLandmarkerOptions(
   base_options=BaseOptions(model_asset_path=hand_model_path),
   running_mode=VisionRunningMode.VIDEO)

pose_options = PoseLandmarkerOptions(
   base_options=BaseOptions(model_asset_path=pose_model_path),
   running_mode=VisionRunningMode.VIDEO)

hand_landmarks = list(range(21)) + list(range(21))
pose_landmarks = [16, 14, 12, 0, 11, 13, 15]

# TODO add x and y for each point
cols = ['h'+str(n) for n in hand_landmarks] + ['p'+str(n) for n in pose_landmarks]
data = pd.DataFrame(columns=cols)

print('DATA LEN', len(data.columns))

def process_video(path, hand_landmarker, pose_landmarker):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            hand_landmarker_result = hand_landmarker.detect_for_video(mp_image, int(time.time()))
            pose_landmarker_result = pose_landmarker.detect_for_video(mp_image, int(time.time()))

            new_row = []

            # process both hands separately
            left_hand = []
            right_hand = []
            for handedness,i in enumerate(hand_landmarker_result.handedness):
                for h_lmk in hand_landmarks:
                    if handedness == 'left':
                        left_hand.append([hand_landmarker_result.hand_landmarks[i][h_lmk].x, 
                                         hand_landmarker_result.hand_landmarks[i][h_lmk].y])
                    else:
                        right_hand.append([hand_landmarker_result.hand_landmarks[i][h_lmk].x,
                                           hand_landmarker_result.hand_landmarks[i][h_lmk].y])

            # if only one hand is present, set the other to all zeroes
            if len(left_hand) == 0:
                left_hand = [0] * 42
            if len(right_hand) == 0:
                right_hand = [0] * 42

            new_row += left_hand + right_hand

            if len(pose_landmarker_result.pose_landmarks) > 0:
                for p_lmk in pose_landmarks:
                    new_row.append([pose_landmarker_result.pose_landmarks[0][p_lmk].x,
                                    pose_landmarker_result.pose_landmarks[0][p_lmk].y])
            else:
                new_row.append([0] * 14)

            print(len(new_row))
            data.loc[len(data)] = new_row

            
            

if __name__ == '__main__':
    with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        with PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
            path = '../WLASL/start_kit/videos/00414.mp4'

            process_video(path, hand_landmarker, pose_landmarker)