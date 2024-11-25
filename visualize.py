import mediapipe as mp
import cv2
import os
import pandas as pd
import json
from tqdm import tqdm
import sys
from argparse import ArgumentParser
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np
import logging

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

# 91 total landmarks
# 42 landmarks per hand (84 total)
hand_landmarks = list(range(21))
# 7 pose landmarks
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

def draw_landmarks_on_image_pose(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def draw_landmarks_on_image_hand(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

  return annotated_image

def process_video(path, hand_landmarker, pose_landmarker, global_frame_num):
    data = pd.DataFrame(columns=cols)
    cap = cv2.VideoCapture(path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0

    if not cap.isOpened():
        logger.warning(f'Skipped {path}')
        return data, 0

    while cap.isOpened():
        ret, frame = cap.read()

        if frame_num >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            break
        
        if not ret:
            tqdm.write(f'[WARN] Quit before frame limit reached on {path}')
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        hand_landmarker_result = hand_landmarker.detect_for_video(mp_image, global_frame_num)
        pose_landmarker_result = pose_landmarker.detect_for_video(mp_image, global_frame_num)

        if frame_num == 30:
            img = draw_landmarks_on_image_pose(frame, pose_landmarker_result)
            img = draw_landmarks_on_image_hand(img, hand_landmarker_result)
            cv2.imwrite('media/drawn.jpg', img)
            cv2.imwrite('media/base.jpg', frame)
            return

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
            logger.error('ERR: improper data format')
            logger.error(f'{len(cols)} | {len(new_row)} ({len(left_hand)}, {len(right_hand)})')
            logger.error(str(new_row))
            sys.exit(0)

        frame_num += 1
        global_frame_num += 1

    return data, frame_num

def generate_lookups(json_path, gloss_to_id_path='gloss-to-id.json', id_to_gloss_path='id-to-gloss.json'):
    gloss_to_id = dict()
    id_to_gloss = dict()

    with open(json_path, 'r') as f:
        json_data = json.load(f)
        for entry in json_data:
            gloss = entry['gloss']
            for instance in entry['instances']:
                id = instance['video_id']

                gloss_to_id[gloss] = gloss_to_id.get(gloss, []) + [id]
                id_to_gloss[id] = gloss

    # save id to gloss dictionary
    try:
        with open(id_to_gloss_path, 'w') as f:
            json.dump(json.dumps(id_to_gloss), f)
    except:
        logger.error(f'{id_to_gloss_path} not saved.')

    # save gloss to id dictionary
    try:
        with open(gloss_to_id_path, 'w') as f:
            json.dump(json.dumps(gloss_to_id), f)
    except:
        logger.error(f'{gloss_to_id_path} not saved.')

            

# Logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--videos-path', '-v', help='Path to video files')
    args = parser.parse_args()

    videos_to_process = sorted([f for f in os.listdir(args.videos_path) if os.path.splitext(f)[1] == '.mp4'])

    # Process videos
    global_frame_num = 0
    processed_videos = pd.DataFrame(columns=['id', 'gloss'])
    with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        with PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
            for video_file in tqdm(videos_to_process[20:]):
                num = os.path.splitext(video_file)[0]

                process_video(f'{args.videos_path}/{video_file}',
                                hand_landmarker, pose_landmarker,
                                global_frame_num)
                
                break