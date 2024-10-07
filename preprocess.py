import mediapipe as mp
import cv2
import os
import pandas as pd
import json
from tqdm import tqdm
import sys

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

def log_video(json, num):
    for entry in json:
        gloss = entry['gloss']
        for instance in entry['instances']:
            if instance['video_id'] == str(num):
                return [num, gloss]


def process_video(path, hand_landmarker, pose_landmarker, global_frame_num):
    data = pd.DataFrame(columns=cols)
    cap = cv2.VideoCapture(path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0

    if not cap.isOpened():
        return data, 0

    while cap.isOpened():
        ret, frame = cap.read()

        if frame_num >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            break
        
        if not ret:
            tqdm.write(f'[WARN] Quit before frame limit reached on {path}')
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        #frame_time = int(global_frame_num * float(1/FPS) * 1000)

        hand_landmarker_result = hand_landmarker.detect_for_video(mp_image, global_frame_num)
        pose_landmarker_result = pose_landmarker.detect_for_video(mp_image, global_frame_num)

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
            tqdm.write('ERR: improper data format')
            tqdm.write(f'{len(cols)} | {len(new_row)} ({len(left_hand)}, {len(right_hand)})')
            tqdm.write(str(new_row))
            sys.exit(0)

        frame_num += 1
        global_frame_num += 1

    return data, frame_num

            
            

if __name__ == '__main__':
    with open('../WLASL/start_kit/WLASL_v0.3.json', 'r') as f:
        json_data = json.load(f)

    global_frame_num = 0

    video_logs = pd.DataFrame(columns=['id', 'gloss'])
    videos_to_process = sorted([f for f in os.listdir('../WLASL/start_kit/videos/') if os.path.splitext(f)[1] == '.mp4'])
    with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        with PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
            for path in tqdm(videos_to_process[:100]):
                num = os.path.splitext(path)[0]

                data, num_frames_processed = process_video(f'../WLASL/start_kit/videos/{path}',
                                                           hand_landmarker, pose_landmarker,
                                                           global_frame_num)
                data.to_csv(f'asl-data/{num}.csv', index=False)

                video_logs.loc[len(video_logs)] = log_video(json_data, num)

                global_frame_num += num_frames_processed
            
            video_logs.to_csv('video-metadata.csv', index=False)