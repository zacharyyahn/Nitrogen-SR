from __future__ import print_function
import os
import time
import shutil
import json
import argparse
import numpy as np
import soundfile as sf
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import time
# import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


import dataset
from utils.params import Params
#from utils.plotting import plot_training

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_name",
    type=str,
    help="The specific file, e.g. ninab1"
)
parser.add_argument(
    "sample_size",
    type=int,
    help="Number of random samples of the test set"
)
args = parser.parse_args()

#Fetch model hyperparameters
checkpoint_dir = "saved_models"
params = Params(checkpoint_dir + "/" + args.model_name + "_hparams.yaml", "DEFAULT")


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Using GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

model_import = __import__('.'.join(['models', params.model]),  fromlist=['object'])
model = model_import.net(params).to(device)
model.load_state_dict(torch.load("saved_models/" + args.model_name + ".ckpt"))

val = model_import.val

#Load model, train function, eval function
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

#Prepare data
test_data = dataset.NO2Dataset(params, "../Dataset/Formatted",  split="test", do_augment=False)
num_epochs = params.epochs

test_sampler = RandomSampler(test_data, num_samples=args.sample_size)

test_loader = DataLoader(
        test_data, 
        batch_size=params.batch_size,
        sampler=test_sampler
    )
total_params = sum(p.numel() for p in model.parameters())

print("Preparing to test", args.model_name)
print("Model size: ", total_params)

#Validation
loss, scores = val(model, device, test_loader, loss_function)
print('Test loss: %.6f, Test Avg. PSNR: %.4f, Test Avg. MSE: %0.6f, Test Avg. SSIM: %0.6f' % (loss, scores["psnr"], scores["mse"], scores["ssim"]))
    