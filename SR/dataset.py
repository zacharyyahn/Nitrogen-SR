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
from skimage.transform import rescale, resize, downscale_local_mean

class NO2Dataset(Dataset):
    def __init__(self, params, img_dir, split="train", flatten=True, do_augment=False):
        """
        data_dir (str): Path to data containing data and labels. 
        X_filename (str): Name of file containing input data. 
        y_filename (str): Name of file containing labels.
        """
        self.dir = img_dir
        self.im_list = []
        self.scale = params.scale

        #If we only want to use one channel, then we have 1/12 as many images
        if params.channel_subset:
            num_images = int(len(params.channel_select)* len(os.listdir(self.dir)) / 12)
            for channel in params.channel_select:
                if split == "train":
                    self.im_list += [file for file in os.listdir(self.dir) if file[file.find("_")+1:] == str(channel) + ".png"][:int(0.6 * 1 / len(params.channel_select) * num_images)]
                if split == "val":
                    self.im_list += [file for file in os.listdir(self.dir) if file[file.find("_")+1:] == str(channel) + ".png"][int(0.6 * 1 / len(params.channel_select) * num_images): int(0.8 * 1 / len(params.channel_select) * num_images)]
                if split == "test":
                    self.im_list += [file for file in os.listdir(self.dir) if file[file.find("_")+1:] == str(channel) + ".png"][int(0.8 * 1 / len(params.channel_select) * num_images):]

        else:
            num_images = len(os.listdir(self.dir))
            self.im_list = os.listdir(self.dir)
            self.SEED = 2023
            random.Random(self.SEED).shuffle(self.im_list)
             #This split method does have a problem of potentially putting multiple filters into different splits. But that should be fine for this preliminary test
            if split == "train":
                self.im_list = self.im_list[:int(0.6 * num_images)]
            if split == "val":
                self.im_list = self.im_list[int(0.6 * num_images): int(0.8 * num_images)]
            if split == "test":
                self.im_list =self.im_list[int(0.8 * num_images):]

        print("Prepared %s dataset of %d images" % (split, len(self.im_list)))

    def __getitem__(self, index):
        hr = Image.open(self.dir + "/" + self.im_list[index])

        #Create the lr by resampling the hr
        hr = np.array(hr).astype(np.float32)
        lr = resize(hr, (hr.shape[0] // self.scale, hr.shape[1] // self.scale))

        #Normalize between 0 and 1. When they were saved they were between 0 and 255
        lr = lr / 255.0
        hr = hr / 255.0
        lr = lr.reshape(3, lr.shape[0], lr.shape[1])
        hr = hr.reshape(3, hr.shape[0], hr.shape[1])
        # print("LR has size", lr.shape) #(100, 100)
        # print("HR has size", hr.shape) #(200, 200)
        return lr, hr

    def __len__(self):
        return len(self.im_list)

