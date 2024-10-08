import mediapipe as mp
import cv2
import os
import pandas as pd
import json
from tqdm import tqdm
import sys
import torch
from RNN import RNN

hand_model_path = 'models/hand_landmarker.task'
pose_model_path = 'models/pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode    

hand_options = HandLandmarkerOptions(
   base_options=BaseOptions(model_asset_path=hand_model_path),
   running_mode=VisionRunningMode.VIDEO,
   num_hands=2,
   )

pose_options = PoseLandmarkerOptions(
   base_options=BaseOptions(model_asset_path=pose_model_path),
   running_mode=VisionRunningMode.VIDEO
   )

hand_landmarks = list(range(21))
pose_landmarks = [16, 14, 12, 0, 11, 13, 15]


def build_cols():
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

    return cols

cols = build_cols()

def start_video(path, hand_landmarker, pose_landmarker, model):
    sequence_length = 0
    data = pd.DataFrame(columns=cols)
    cap = cv2.VideoCapture(path)
    FPS = cap.get(cv2.CAP_PROP_FPS)

    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        hand_landmarker_result = hand_landmarker.detect_for_video(mp_image, frame_num)
        pose_landmarker_result = pose_landmarker.detect_for_video(mp_image, frame_num)

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
        if len(left_hand) == 0 and len(right_hand) != 84:
            left_hand = [0] * 42
        if len(right_hand) == 0 and len(left_hand) != 84:
            right_hand = [0] * 42

        new_row += left_hand + right_hand

        if len(pose_landmarker_result.pose_landmarks) > 0:

            for p_lmk in pose_landmarks:
                new_row += [pose_landmarker_result.pose_landmarks[0][p_lmk].x,
                            pose_landmarker_result.pose_landmarks[0][p_lmk].y]
        else:
            new_row += [0] * 14

        try:
            data.loc[len(data)] = new_row
        except:
            print('ERR: improper data format')
            print(f'{len(cols)} | {len(new_row)} ({len(left_hand)}, {len(right_hand)})')
            print(str(new_row))
            sys.exit(0)

        frame_num += 1

        if len(data) > sequence_length:
            data = data.drop(axis=0, index=0)
        if len(data) == sequence_length:
            pred = model(data)
            print(pred)
            
            

if __name__ == '__main__':
    model = RNN()
    model.load_state_dict(torch.load('rnn_asl.pth', weights_only=True))

    with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        with PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
                start_video(hand_landmarker, pose_landmarker, model)