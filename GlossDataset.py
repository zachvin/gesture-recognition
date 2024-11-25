from torch.utils.data import Dataset
import os
import torch
import pandas as pd

class GlossDataset(Dataset):
  def __init__(self, annotations_file, landmark_dir, sequence_length):
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

    self.gloss_and_id = pd.read_csv(annotations_file, dtype={'id':'object', 'gloss':'category'})
    cat_map = {gloss_name: gloss_id for gloss_id, gloss_name in enumerate(self.gloss_and_id['gloss'].cat.categories)}
    self.gloss_and_id['gloss'] = self.gloss_and_id['gloss'].replace(cat_map)
    self.gloss_and_id['gloss'] = self.gloss_and_id['gloss'].astype(int)

    self.landmark_dir = landmark_dir
    self.sequence_length = sequence_length
    self.num_gestures = self.gloss_and_id['gloss'].nunique()

  def __len__(self):
    return len(self.gloss_and_id)

  def __getitem__(self, idx):
    landmark_path = os.path.join(self.landmark_dir, self.gloss_and_id['id'].iloc[idx] + '.csv')

    gloss = self.gloss_and_id['gloss'].iloc[idx]

    landmarks = pd.read_csv(landmark_path)
    # pad output to make video long enough
    if landmarks.shape[1] - self.sequence_length > 0:
      delta = landmarks.shape[1] - self.sequence_length
      row = landmarks.iloc[-1]
      for _ in range(delta):
        landmarks.loc[len(landmarks)] = row

    # trim output if it's too long
    landmarks_tensor = torch.tensor(landmarks.iloc[:self.sequence_length].to_numpy().astype('float32'))
    labels_tensor = torch.tensor(gloss, dtype=torch.long)

    return landmarks_tensor, gloss
