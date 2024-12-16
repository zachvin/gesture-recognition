from torch.utils.data import Dataset
import os
import torch
import pandas as pd
import numpy as np
import random

class GlossDataset(Dataset):
  def __init__(self, annotations_file, landmark_dir, sequence_length, demo=False):
    """
    Params:
        annotations_file (str): Path to CSV with id,gloss.
        landmark_dir (str): Path to directory where preprocessed video landmarks are stored.
        sequence_length (int): Number of frames to retrieve from each video. Smaller videos are padded
            and larger videos are truncated.

    Returns:
        GlossDataset object.
    """
    super(GlossDataset, self).__init__()

    self.demo = demo

    self.gloss_and_id = pd.read_csv(annotations_file, dtype={'id':'object', 'gloss':'category'})
    cat_map = {gloss_name: int(gloss_id) for gloss_id, gloss_name in enumerate(self.gloss_and_id['gloss'].cat.categories)}
    self.gloss_and_id['gloss'] = self.gloss_and_id['gloss'].cat.rename_categories(cat_map)

    self.landmark_dir = landmark_dir
    self.sequence_length = sequence_length
    self.num_gestures = self.gloss_and_id['gloss'].nunique()
    self.all_ids = []

    self.loaded_data = []
    for idx in range(len(self.gloss_and_id)):
        landmark_path = os.path.join(self.landmark_dir, self.gloss_and_id['id'].iloc[idx] + '.csv')
        try:
            landmarks = pd.read_csv(landmark_path)
            self.all_ids.append(self.gloss_and_id['id'].iloc[idx])
        except FileNotFoundError:
            if self.demo:
                continue
            raise FileNotFoundError
        self.loaded_data.append(landmarks)

  def __len__(self):
    return len(self.loaded_data)

  def __getitem__(self, idx):
    gloss = self.gloss_and_id['gloss'].iloc[idx]
    landmarks = self.loaded_data[idx]

    # pad output to make video long enough
    if landmarks.shape[0] < self.sequence_length:
      delta = self.sequence_length - landmarks.shape[0]
      last_row = landmarks.iloc[-1]
      for _ in range(delta):
        landmarks.loc[len(landmarks)] = last_row

    # trim output if it's too long
    if self.demo:
        gloss = self.gloss_and_id[self.gloss_and_id['id'] == self.all_ids[idx]]['gloss'].values[0]
    landmarks_tensor = torch.tensor(landmarks.iloc[:self.sequence_length].to_numpy().astype('float32'))
    labels_tensor = torch.tensor(gloss, dtype=torch.long)

    return landmarks_tensor, labels_tensor, self.all_ids[idx]

class NoiseWrapper(Dataset):
    def __init__(self, base_dataset, noise_level=0.05):
        super(NoiseWrapper, self).__init__()
        self.base_dataset = base_dataset
        self.noise_level = noise_level
        self.factor = 0.01

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # dilation_factor = (random.random() - 0.5) * 0.3 # makes dilation factor from -0.15 to 0.15
        landmarks, gloss, _ = self.base_dataset[idx]

        # Occasionally add noise to increase size of dataset
        #if random.random() > 0.5:
        landmarks += np.random.normal(0, self.noise_level, landmarks.shape)

        # Occasionally scale landmarks
        if random.random() > 0.5:
            landmarks += landmarks * ((random.random() * self.factor) - (self.factor / 2))
            landmarks = np.clip(landmarks, -1, 1)
            
        return landmarks.to(torch.float32), gloss, _


