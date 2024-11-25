import mediapipe as mp
import cv2
import os
import pandas as pd
import json
from tqdm import tqdm
import sys
import torch
from Models import RNN, LSTM

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

int_to_gloss = {
    0: 'book',
    1: 'computer',
    2: 'backpack',
    3: 'medicine',
    4: 'teacher',
}

def start_video(hand_landmarker, pose_landmarker, model):
    sequence_length = 50
    data = pd.DataFrame(columns=cols)
    cap = cv2.VideoCapture(0)
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
            data = data.drop([0])
        if len(data) == sequence_length:
            data_tensor = torch.tensor(data.to_numpy())
            pred = model(data_tensor).detach().numpy().argmax()
            print(int_to_gloss[pred])

        cv2.imshow('Preview', frame)

        k = cv2.waitKey(1)
        if k == 27:
            break
            
            
input_size = 98 # 7 landmarks for upper body and 21 for each hand for a total of
                # 49 landmarks * 2 x/y positions for each
sequence_length = 50 # 25 fps, assuming about two seconds per video
num_layers = 2
hidden_size = 128
num_classes = 5 # number of signs
batch_size = 1

if __name__ == '__main__':
    model = RNN(input_size, hidden_size, num_layers, num_classes, batch_size)
    model.load_state_dict(torch.load('models/asl_rnn.pth', map_location=torch.device('cpu'), weights_only=True))

    with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        with PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
                start_video(hand_landmarker, pose_landmarker, model)

    print('Closing...')