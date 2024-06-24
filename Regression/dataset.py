#This dataset normalizes each image between 0 and 1 and does the same with n02
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
import cv2
import time
from skimage.transform import rescale, resize, downscale_local_mean

class NO2Dataset(Dataset):
    def __init__(self, params, img_dir, split, flatten=True, do_augment=False):
        """
        data_dir (str): Path to data containing data and labels. 
        X_filename (str): Name of file containing input data. 
        y_filename (str): Name of file containing labels.
        """
        self.dir = img_dir
        self.data = pd.read_csv("../Dataset/Raw/samples_S2S5P_monthly_epa.csv")
        self.data.dropna(axis=0, inplace=True, thresh=7)
        self.idx_list = list(self.data["idx"])
        self.SEED = 2023
        self.num_channels = params.num_inputs
        random.Random(self.SEED).shuffle(self.idx_list)

        num_images = len(self.idx_list)
        if split == "train":
            self.idx_list = self.idx_list[:int(0.6 * num_images)]
        if split == "val":
            self.idx_list = self.idx_list[int(0.6 * num_images): int(0.8 * num_images)]
        if split == "test":
            self.idx_list =self.idx_list[int(0.8 * num_images):]

        print("Prepared %s dataset of %d images" % (split, len(self.idx_list)))

    def __getitem__(self, index):
        path = self.data["AirQualityStation"][self.idx_list[index]]
        n02 = self.data["no2"][self.idx_list[index]]

        channels = [im_path for im_path in os.listdir(self.dir) if im_path[:im_path.find("_")] == (str(self.idx_list[index]) + "-" + path)]

        im = cv2.imread(self.dir + channels[0])
        ims = []
        for path in channels:
            this_im = cv2.imread(self.dir + path)
            this_im = this_im[:, :, 0]
            ims.append(this_im)
        im = np.stack(ims)
        im = im[:self.num_channels, :, :]


        #Note: Images are already normalized when they are saved, they just need to be rescaled from 0-255 to 0-1
        im = im.astype('float64')
        im /= 255.0
        im = (im - np.mean(im)) / np.std(im)
        # im = im[:4, :, :]
        
        #Normalize between 0 and 1 - values are computed in separate spreadsheet
        max_n02 = 71.75532137
        min_n02 = -2.68378695
        avg_n02 = 18.71056311
        std_n02 = 13.1422108
        #n02 = (n02 - min_n02) / (max_n02 - min_n02)
        n02 = (n02 - avg_n02)/std_n02
        return im, n02

    def __len__(self):
        return len(self.idx_list)

