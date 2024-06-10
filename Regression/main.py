from __future__ import print_function
import os
import time
import shutil
import json
import argparse
import numpy as np
import soundfile as sf
import shutil
import torch.optim.lr_scheduler as lr_scheduler

import torch
import torch.nn as nn
import torch.optim as optim
import time
# import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


import dataset as dataset
from utils.params import Params
#from utils.plotting import plot_training

#Fetch model hyperparameters
params = Params("hparams.yaml", "DEFAULT")
checkpoint_dir = "saved_models"

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Using GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")

#save the params for reproducibility, version number increases separately for each transform type
version = int(len([file for file in os.listdir("saved_models") if file[:file.find("_")] == params.model]) / 2)
shutil.copy("hparams.yaml", checkpoint_dir + "/" + params.model + "_v" + str(version) + "_hparams.yaml")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

model_import = __import__('.'.join(['models', params.model]),  fromlist=['object'])
model = model_import.net(params).to(device).double()

train = model_import.train
val = model_import.val

#Load model, train function, eval function
loss_function = torch.nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, weight_decay=0.00005)
#scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=15)

#Prepare data
train_data = dataset.NO2Dataset(params, params.data_path,  split="train", do_augment=True)
val_data = dataset.NO2Dataset(params, params.data_path, split="val", do_augment=True)
num_epochs = params.epochs

train_loader = DataLoader(
        train_data, 
        batch_size=params.batch_size,
        #shuffle=True,
    )
val_loader = DataLoader(
        val_data,
        batch_size=params.batch_size, #we always use a batch size of 10 so that we can combine the fragments
        #shuffle=False,
    )

print("Preparing to train", params.model, "for", num_epochs, "epochs.")
print("Batch size:", params.batch_size, ", lr:", params.lr, ", loss_fn: MSE")
print("Images read from", params.data_path)
global_start = time.perf_counter()

#Training and validation
train_losses, train_psnrs, val_losses, val_psnrs = [], [], [], []
for epoch in range(num_epochs):
    start_time = time.perf_counter()
    print("---- Epoch Number: %s ----" % (epoch + 1))

    #Train
    train(model, device, train_loader, optimizer, loss_function)
        # Evaluate on both the training and validation set. 
    train_loss, train_scores = val(model, device, train_loader, loss_function)
    print('\rEpoch: [%d/%d], Train loss: %.6f' % (epoch+1, num_epochs, train_loss))

    #scheduler.step()

    #Validation
    val_loss, val_scores = val(model, device, val_loader, loss_function)
    print('Epoch: [%d/%d], Valid loss: %.6f' % (epoch+1, num_epochs, val_loss))
    
    # Collect some data for logging purposes. 
    train_losses.append(float(train_loss))
    val_losses.append(float(val_loss))
    end_time = time.perf_counter()
    print("Took %0.1f seconds" % (end_time - start_time))

    #If we've found the best model so far
    if val_loss == np.min(val_losses):

        print("Found the current best model for this training, saving....")

        global_end = time.perf_counter()
        torch.save(model.state_dict(), checkpoint_dir + "/" + params.model +"_v" + str(version) + ".ckpt")
            
    plt.clf()   
    plt.cla() 
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss (" + "MSE" + ")")
    plt.savefig("figures/" + params.model + "_v" + str(version) + "_loss.png")

        # logs ={
        #     "model": params.model,
        #     "best_val_epoch": int(np.argmax(val_losses)+1),
        #     "lr": params.lr,
        #     "batch_size":params.batch_size,
        #     "start_time":global_start,
        #     "end_time":global_end,
        #     "train_losses": train_losses,
        #     "train_mse": train_mses,
        #     "val_losses": val_losses,
        #     "val_mse": val_mses,
        #     "notes": "Added RNN dropout of 0.3"
        # }

        # with open(checkpoint_dir + "/" + params.model +"_v" + str(version) + "_logs.json", 'w') as f:
        #     json.dump(logs, f)
    
plt.clf()    
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("figures/" + params.model + "_v" + str(version) + "_loss.png")

    # Save model
    # valid_losses.append(valid_loss.item())
    # if np.argmin(valid_losses) == epoch:
    #     print('Saving the best model at %d epochs!' % epoch)
    #     torch.save(cnn.state_dict(), 'best_model.ckpt')
#Log