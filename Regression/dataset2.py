#In this dataset we use the raw npy files
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
    def __init__(self, params, img_dir, split="train", flatten=True, do_augment=False):
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
        random.Random(self.SEED).shuffle(self.idx_list)

        num_images = len(self.idx_list)
        num_images = 2 / 0.6
        if split == "train":
            self.idx_list = self.idx_list[:int(0.6 * num_images)]
        if split == "val":
            self.idx_list = self.idx_list[int(0.6 * num_images): int(0.8 * num_images)]
        if split == "test":
            self.idx_list =self.idx_list[int(0.8 * num_images):]

        print("Prepared %s dataset of %d images" % (split, len(self.idx_list)))

    def __getitem__(self, index):
        path = self.data["img_path"][self.idx_list[index]]
        n02 = self.data["no2"][self.idx_list[index]]

        im = np.load("../Dataset/Raw/sentinel-2/"+path)

        #Means and stdevs computed in https://github.com/HSG-AIML/Global-NO2-Estimation/blob/main/satellite_model/transforms.py
        self.channel_means = np.array([340.76769064, 429.9430203, 614.21682446,
                590.23569706, 950.68368468, 1792.46290469, 2075.46795189, 2218.94553375,
                2266.46036911, 2246.0605464, 1594.42694882, 1009.32729131])

        self.channel_std = np.array([554.81258967, 572.41639287, 582.87945694,
                675.88746967, 729.89827633, 1096.01480586, 1273.45393088, 1365.45589904,
                1356.13789355, 1302.3292881, 1079.19066363, 818.86747235])
        
        self.no2_mean = 20.95862054085057
        self.no2_std = 11.641219387279973

        #rearrange the channels to match BigEarthNet pretraining
        reordered_img = np.zeros(im.shape)
        reordered_img[0, :, :] = im[10, :, :]
        reordered_img[1, :, :] = im[2, :, :]
        reordered_img[2, :, :] = im[1, :, :]
        reordered_img[3, :, :] = im[0, :, :]
        reordered_img[4, :, :] = im[4, :, :]
        reordered_img[5, :, :] = im[5, :, :]
        reordered_img[6, :, :] = im[6, :, :]
        reordered_img[7, :, :] = im[3, :, :]
        reordered_img[8, :, :] = im[7, :, :]
        reordered_img[9, :, :] = im[11, :, :]
        reordered_img[10, :, :] = im[8, :, :]
        reordered_img[11, :, :] = im[9, :, :]

        #Normalize on Gaussian(0, 1)
        for i in range(12):
            reordered_img[i, :, :] = (reordered_img[i, :, :] - self.channel_means[i]) / self.channel_std[i]
        
        reordered_img = reordered_img.reshape((12, 200, 200))

        #Normalize with mean and stdev
        #reordered_img = (reordered_img - np.min(reordered_img)) / (np.max(reordered_img) - np.min(reordered_img))
        
        n02 = (n02 - self.no2_mean) / self.no2_std
        #reordered_img = (reordered_img - np.mean(reordered_img) / np.std(reordered_img))
        return reordered_img, n02

    def __len__(self):
        return len(self.idx_list)

