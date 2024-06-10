#Can do augmentation by saving a type of augmentation with each file in the list and then doing the corresponding augmentation at retrieval time so that you don't have to hold it all in memory
import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import random
from torch.utils.data.dataset import Dataset
from PIL import Image
import time
import cv2
from skimage.transform import rescale, resize, downscale_local_mean

#Closest bands
#20m   10m
#5     4
#6     4
#7     8
#8a    8
#11    8
#12    8

class NO2Dataset(Dataset):
    def __init__(self, params, img_dir, split="train", flatten=True, do_augment=False):
        self.dir = img_dir
        self.data = pd.read_csv("../Dataset/Raw/samples_S2S5P_monthly_epa.csv")
        self.data.dropna(axis=0, inplace=True, thresh=7)
        self.idx_list = list(self.data["idx"])
        self.img_pairs = []
        for path in os.listdir(img_dir):
            if path[path.find("_"):] in ["_4.png", "_5.png"]: #bands 5, 6
                self.img_pairs.append([path, path[:path.find("_")] + "_0.png"]) #pair with band 4
            elif path[path.find("_"):] in ["_6.png", "_7.png", "_8.png", "_9.png"]: #bands 7, 8a, 11, 12
                self.img_pairs.append([path, path[:path.find("_")] + "_3.png"]) #pair with band 8

        self.SEED = 2023
        random.Random(self.SEED).shuffle(self.img_pairs)

        num_images = len(self.img_pairs)
        if split == "train":
            self.img_pairs = self.img_pairs[:int(0.6 * num_images)]
        if split == "val":
            self.img_pairs = self.img_pairs[int(0.6 * num_images): int(0.8 * num_images)]
        if split == "test":
            self.img_pairs =self.img_pairs[int(0.8 * num_images):]

        print("Prepared %s dataset of %d images" % (split, len(self.img_pairs)))

    def __getitem__(self, index):
        pair = self.img_pairs[index]
        lr_path = pair[0]
        hr_path = pair[1]
        lr = cv2.imread(self.dir + "/" + lr_path).astype('float64')
        hr = cv2.imread(self.dir + "/" + hr_path).astype('float64')

        #Normalize between 0 and 1. When they were saved they were between 0 and 255
        lr = lr / 255.0
        hr = hr / 255.0
        lr = lr.reshape(3, lr.shape[0], lr.shape[1])
        hr = hr.reshape(3, hr.shape[0], hr.shape[1])
        # print("LR has size", lr.shape) #(100, 100)
        # print("HR has size", hr.shape) #(200, 200)
        return lr, hr

    def __len__(self):
        return len(self.img_pairs)

