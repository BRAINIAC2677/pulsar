from config import CFG
import sklearn.preprocessing
import numpy as np
from ast import literal_eval as make_tuple
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from utils import adj_mat

def df_to_numpy(df):
  print("DF TO NUMPY")
  x_elems = df.drop("LABEL", axis=1).values
  print("X ELEMS SHAPE: ", x_elems.shape)
  x = [make_tuple(elem) for elem in x_elems for elem in elem]
  x = np.array(x)
  print("X SHAPE: ", x.shape)
  x = x.reshape(len(x_elems), 3*21)
  print("X SHAPE: ", x.shape)
  lb = sklearn.preprocessing.LabelBinarizer().fit(CFG.classes)
  y = np.array(df["LABEL"]).reshape(-1,1)
  print("Y SHAPE: ", y.shape)
  y_ohe = lb.transform(y)
  y_ohe = np.hstack((y_ohe,1-y_ohe)) ### ADJUSTMENT FOR CLASS = 2
  print("Y OHE SHAPE: ", y_ohe.shape)

  train_data_numpy = (x, y_ohe)
  return train_data_numpy


class HandPoseDatasetMapped(Dataset):
    def __init__(self, data_list, joint_stream=CFG.joint_stream,bone_stream=CFG.bone_stream, vel_stream=CFG.vel_stream, acc_stream=CFG.acc_stream):
        self.data_list = data_list # List of tuples: [(x1, y1), (x2, y2), ...]
        self.joint_stream=joint_stream
        self.bone_stream = bone_stream
        self.vel_stream = vel_stream
        self.acc_stream = acc_stream
        self.index_map = [] 
        
        # Build the Index Map
        for patient_idx, (x_data, _) in enumerate(self.data_list):
            num_frames = len(x_data)
            # Calculate how many valid sequences this patient has
            num_sequences = num_frames - CFG.sequence_length + 1
            
            if num_sequences > 0:
                for start_frame in range(num_sequences):
                    # Store the address: (Which Patient?, Which Frame?)
                    self.index_map.append((patient_idx, start_frame))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 1. Look up where the data lives
        patient_idx, start_frame = self.index_map[idx]
        
        # 2. Retrieve the specific patient's data
        full_x, full_y = self.data_list[patient_idx]
        
        # 3. Slice safely
        end_frame = start_frame + CFG.sequence_length
        seq_x = full_x[start_frame : end_frame]
        seq_y = full_y[start_frame : end_frame]

        # Reshape and feature engineering...
        x = seq_x.reshape(CFG.sequence_length, 21, 3)
        
        if CFG.add_feats:
            #additional features: https://link.springer.com/content/pdf/10.1186/s13640-019-0476-x.pdf
            #r = np.sqrt(x**2 + y**2 + z**2)
            #theta = np.arccos(z/r)
            #phi = np.arctan(y/x)
            theta = phi = r = np.zeros_like(x[:, :, 0])

            if x.all() != 0:
              r = np.sqrt(x[:, :, 0]**2 + x[:, :, 1]**2 + x[:, :, 2]**2)
              theta = np.arccos(x[:, :, 2]/r)
              phi = np.arctan(x[:, :, 1]/x[:, :, 0])
            add_feats = np.stack([r, theta, phi], axis=-1)
            
            x = np.concatenate((x, add_feats),axis=2)

        if self.bone_stream:
            dist = np.zeros_like(x)
            for v1, v2 in adj_mat.inward:
                dist[:, v1, :] = x[:, v2, :] - x[:, v1, :]
            x = dist

        if self.vel_stream:
            dist = np.zeros_like(x)
            for v1, v2 in adj_mat.inward:
                dist[:, v1, :] = x[:, v2, :] - x[:, v1, :]
            vel = np.gradient(dist, axis=0)
            x = vel
            
        if self.acc_stream:
            dist = np.zeros_like(x)
            for v1, v2 in adj_mat.inward:
                dist[:, v1, :] = x[:, v2, :] - x[:, v1, :]
            vel = np.gradient(dist, axis=0)
            acc = np.gradient(vel, axis=0)
            x = acc

        if CFG.add_phi:
            phi = np.arctan(x[:, :, 1]/x[:, :, 0])
            x = np.concatenate((x, phi),axis=2)

        
        # [Insert Stream Logic Here]

        return x, seq_y