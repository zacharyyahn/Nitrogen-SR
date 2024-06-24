from __future__ import print_function
import os
import time
import shutil
import sys
import json
import argparse
import numpy as np
#import soundfile as sf
import shutil
import torch.optim.lr_scheduler as lr_scheduler

import torch
import torch.nn as nn
import torch.optim as optim
import time
# import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

#from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


import dataset as dataset
from params import Params
#from utils.plotting import plot_training

#Fetch model hyperparameters
params = Params("hparams.yaml", "DEFAULT")
checkpoint_dir = "saved_models"

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Using GPU", file=sys.stderr)
else:
    device = torch.device('cpu')
    print("Using CPU", file=sys.stderr)

#save the params for reproducibility, version number increases separately for each transform type
version = int(len([file for file in os.listdir("saved_models") if file[:file.find("_")] == params.model]) / 2)
shutil.copy("hparams.yaml", checkpoint_dir + "/" + params.model + "_v" + str(version) + "_hparams.yaml")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

lrs = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
batch_sizes = [1, 2, 4, 8, 16]
results = {}
for lr in lrs:
    for batch_size in batch_sizes:

        model_import = __import__('.'.join(['models', params.model]),  fromlist=['object'])
        model = model_import.net(params).to(device).double()

        train = model_import.train
        val = model_import.val

        #Load model, train function, eval function
        loss_function = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #scheduler = lr_scheduler.StepLR(optimizer, 10, 0.5, -1)
        #optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, weight_decay=0.00005)
        #scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=15)

        #Prepare data
        train_data = dataset.NO2Dataset(params, params.data_path,  split="train", do_augment=True)
        val_data = dataset.NO2Dataset(params, params.data_path, split="val", do_augment=True)
        num_epochs = params.epochs

        train_loader = DataLoader(
                train_data, 
                batch_size=batch_size,
                #shuffle=True,
            )
        val_loader = DataLoader(
                val_data,
                batch_size=batch_size, #we always use a batch size of 10 so that we can combine the fragments
                #shuffle=False,
            )
        print("================", file=sys.stderr)
        print("================", file=sys.stderr)
        print("Preparing to train", params.model + str(params.resnet_depth), "for", num_epochs, "epochs.", file=sys.stderr)
        print("Batch size:", batch_size, ", lr:", lr, ", loss_fn: MAE", file=sys.stderr)
        print("Images read from", params.data_path, file=sys.stderr)
        print("Notes:", params.note, file=sys.stderr)
        print("================", file=sys.stderr)
        print("================", file=sys.stderr)
        global_start = time.perf_counter()

        #Training and validation
        train_losses, train_psnrs, val_losses, val_psnrs = [], [], [], []
        for epoch in range(num_epochs):
            start_time = time.perf_counter()
            print("---- Epoch Number: %s ----" % (epoch + 1), file=sys.stderr)
            #print(f"Learning Rate is {scheduler.get_last_lr()}", file=sys.stderr)
            #Train
            train(model, device, train_loader, optimizer, loss_function)
                # Evaluate on both the training and validation set. 
            train_loss, train_scores = val(model, device, train_loader, loss_function)
            print('\rEpoch: [%d/%d], Train loss: %.6f, Real MSE: %0.6f, Real MAE: %0.6f' % (epoch+1, num_epochs, train_loss, train_scores["real_mse"], train_scores["real_mae"]), file=sys.stderr)

            #scheduler.step()

            #Validation
            val_loss, val_scores = val(model, device, val_loader, loss_function)
            print('Epoch: [%d/%d], Valid loss: %.6f, Real MSE: %0.6f, Real MAE: %0.6f' % (epoch+1, num_epochs, val_loss, val_scores["real_mse"], val_scores["real_mae"]), file=sys.stderr)
            
            # Collect some data for logging purposes. 
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            end_time = time.perf_counter()
            print("Took %0.1f seconds" % (end_time - start_time), file=sys.stderr)

            #scheduler.step()    

        results[(lr,batch_size)] = np.min(val_losses)
print(results, file=sys.stderr)
