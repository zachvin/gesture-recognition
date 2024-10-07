import mediapipe as mp
import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

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
   running_mode=VisionRunningMode.VIDEO,
   num_hands=2)

pose_options = PoseLandmarkerOptions(
   base_options=BaseOptions(model_asset_path=pose_model_path),
   running_mode=VisionRunningMode.VIDEO)

hand_landmarks = list(range(21))
pose_landmarks = [16, 14, 12, 0, 11, 13, 15]

cols = []
for n in hand_landmarks:
    for dir in ['x','y']:
        cols.append(f'lh{str(n)}{dir}')

for n in hand_landmarks:
    for dir in ['x','y']:
        cols.append(f'rh{str(n)}{dir}')

for n in pose_landmarks:
    for dir in ['x','y']:
        cols.append(f'p{str(n)}{dir}')


def process_video(path, hand_landmarker, pose_landmarker):
    data = pd.DataFrame(columns=cols)
    cap = cv2.VideoCapture(path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0

    #pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), position=1)
    while cap.isOpened():
        ret, frame = cap.read()

        if frame_num >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            break
        
        if ret:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            frame_time = int(frame_num * (1/FPS) * 1000)

            hand_landmarker_result = hand_landmarker.detect_for_video(mp_image, frame_time)
            pose_landmarker_result = pose_landmarker.detect_for_video(mp_image, frame_time)

            new_row = []

            # process both hands separately
            left_hand = []
            right_hand = []
            for i,handedness in enumerate(hand_landmarker_result.handedness):
                for h_lmk in hand_landmarks:
                    if handedness[0].category_name == 'Left':
                        left_hand += [hand_landmarker_result.hand_landmarks[i][h_lmk].x, 
                                    hand_landmarker_result.hand_landmarks[i][h_lmk].y]
                    else:
                        right_hand += [hand_landmarker_result.hand_landmarks[i][h_lmk].x,
                                    hand_landmarker_result.hand_landmarks[i][h_lmk].y]

            # if only one hand is present, set the other to all zeroes
            if len(left_hand) == 0:
                left_hand = [0] * 42
            if len(right_hand) == 0:
                right_hand = [0] * 42

            new_row += left_hand + right_hand

            if len(pose_landmarker_result.pose_landmarks) > 0:

                for p_lmk in pose_landmarks:
                    new_row += [pose_landmarker_result.pose_landmarks[0][p_lmk].x,
                                pose_landmarker_result.pose_landmarks[0][p_lmk].y]
            else:
                new_row.append([0] * 14)

            data.loc[len(data)] = new_row

            frame_num += 1
            #pbar.update(1)
    
    #pbar.close()
    return data

            
            

if __name__ == '__main__':
    videos_to_process = [f for f in os.listdir('../WLASL/start_kit/videos/') if os.path.splitext(f)[1] == '.mp4']
    with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        with PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
            for path in tqdm(videos_to_process):
                num = os.path.splitext(path)[0]
                data = process_video(path, hand_landmarker, pose_landmarker)
                data.to_csv(f'data/{num}.csv', index=False)
