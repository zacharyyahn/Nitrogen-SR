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
from audiomentations import Compose, AddGaussianNoise, PitchShift, Shift, HighPassFilter, Gain, PolarityInversion

class GTZANDataset(Dataset):
    def __init__(self, params, split="train", flatten=True, do_augment=False):
        """
        data_dir (str): Path to data containing data and labels. 
        X_filename (str): Name of file containing input data. 
        y_filename (str): Name of file containing labels.
        """
        model_params = {
            "mfcc":params.mfcc_width,
            "melspectrogram":params.mel_width,
            "tempogram":params.tempo_width,
            "chroma_cqt":params.chroma_width
        }
        self.do_augment = do_augment
        self.data_width = model_params[params.transform]
        data_dir = params.data_dir
        self.params = params
        self.transform_type = params.transform
        self.SEED = 2023
        self.GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        self.data_dir = data_dir + "genres_original/"
        self.file_list = []   
        self.dataset = []  
        
        for genre in os.listdir(data_dir + "genres_original"):
            for filename in os.listdir(data_dir + "genres_original/" + genre):
                file_name = genre + "/" + filename
                self.file_list.append(file_name)
        
        #now make the split
        random.Random(self.SEED).shuffle(self.file_list)
        if split == "train":
            self.file_list = self.file_list[:600]

        if split == "test":
            self.file_list = self.file_list[600:800]

        if split == "val":
            self.file_list = self.file_list[800:] 

        if split == "train":
            scale_factor = params.data_scale
        else:
            scale_factor = 1
        for file in self.file_list:
            for index in range(1, 11):
                for i in range(scale_factor):
                    self.dataset.append(({"index":index}, file))

        print("Finishing making %s dataset with %s items" % (split, len(self.dataset)))
    
        transforms = [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(p=0.5),
            HighPassFilter(p=0.2),
            Gain(min_gain_db=-12, max_gain_db=12, p=0.5),
            PolarityInversion(p=0.2)
        ]
        self.augmentation = Compose(transforms=transforms)

    def __getitem__(self, index):
        (aug_dict, this_file) = self.dataset[index]
        #this_file = self.dataset[index]
        this_genre = this_file[:this_file.find("/")]
        this_genre_index = self.GENRES.index(this_genre)

        #Read in the wav from the given file name
        wav, fs = sf.read(self.data_dir + this_file)

        #Trim the wav so everything is the same length. Padding was not as effective.
        wav = wav[:660000]

        #Then break the wav into its own chunk
        start = int(len(wav) * (aug_dict["index"] - 1) / 10)
        end = int(len(wav) * (aug_dict["index"] / 10)) 
        wav = wav[start:end]

        if self.do_augment:
            wav = self.augmentation(samples=wav, sample_rate=22050)

        #If an augmentation is assigned to this wav, do that first
        # if aug_dict != {}:
        #     if aug_dict["noise"] != 0:
        #         wav = add_noise(wav, aug_dict["noise"])
        #     if aug_dict["pitch"] != 0:
        #         wav = pitch_shift(wav, aug_dict["pitch"])
        
        #Normalize the data
        #wav = min_max_norm(wav)

        #wav = np.pad(wav, (0, 676000 - len(wav)), mode="wrap").astype(np.float32) #pad to make sure they're all the same size
        #wav = self.compute_mfcc(wav, 1000, 12)
        if self.transform_type == "mfcc":
            wav = librosa.feature.mfcc(y=wav, sr=22050, n_mfcc=self.params.mfcc_width, n_fft=1024).astype(np.float32)
            wav = librosa.power_to_db(wav, ref=np.max)
        elif self.transform_type == "melspectrogram":
            wav = librosa.feature.melspectrogram(y=wav, sr=22050, n_fft=1024).astype(np.float32)
            wav = librosa.power_to_db(wav, ref=np.max)
            wav = wav[:self.params.mel_width, :] #reshape in case we want to trim off the end
        elif self.transform_type == "tempogram":
            wav = librosa.feature.tempogram(y=wav, sr=22050).astype(np.float32)
            wav = wav[:self.params.tempo_width, :] #reshape in case we want to trim off the end
        elif self.transform_type == "chroma_cqt":
            wav = librosa.feature.chroma_cqt(y=wav, sr=22050).astype(np.float32)
            wav = wav[:self.params.mel_width, :] #reshape in case we want to trim off the end
        return wav, this_genre_index

    def __len__(self):
        return len(self.dataset)

def pitch_shift(wav, steps):
    wav = librosa.effects.pitch_shift(wav, sr=20050, n_steps=steps)
    return wav

def add_noise(wav, std_dev):
    noise = np.random.normal(0, std_dev, wav.shape[0])
    wav = wav + noise
    return wav

def min_max_norm(wav):
    return 2 * (np.max(wav) - wav) / (np.max(wav) - np.min(wav)) - 1


