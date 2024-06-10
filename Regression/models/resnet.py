import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet50
import os

class net(nn.Module):
    def __init__(self, params):
        super(net, self).__init__()

        #Create a ResNet50 architecture and change its conv1 layer to accept 12 channel inputs
        self.backbone = resnet50(pretrained=False, num_classes=19)
        self.backbone.conv1 = torch.nn.Conv2d(params.num_inputs, 64, kernel_size=(3,3), stride=(2,2), padding=(3,3), bias=False)

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        
        #Load in pretrained model from LUC dataset, which had 19 classes
        #self.backbone.load_state_dict(torch.load("./saved_models/pretrained_resnet50_LUC.model", map_location=device))

        #Reset the original fc layer
        self.backbone.fc = nn.Identity()

        #Add our own regression head
        self.head = nn.Sequential(nn.Linear(2048, 512), nn.Sigmoid(), nn.Linear(512, 1))

        #Try freezing everything but the head
        for param in self.backbone.parameters():
            param.require_grad = True
        for param in self.head.parameters():
            param.require_grad = True
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def train(model, device, train_loader, optimizer, loss_function):
    model.train()
    for (im, n02) in tqdm(train_loader, leave=False):
        im = im.to(device)
        n02 = n02.to(device)

        #Forward pass
        optimizer.zero_grad()
        out = model(im)
        loss = loss_function(out, n02)

        #print("Expected", n02, "got", out, "propagating loss", loss)

        #Backward pass
        loss.backward()
        optimizer.step()

def val(model, device, loader, loss_function):
    max_n02 = 71.75532137
    min_n02 = -2.68378695
    model.eval()
    losses = []
    real_mses = []
    real_maes = []
    with torch.no_grad():
        for im, n02 in tqdm(loader, leave=False):
            im = im.to(device)
            n02 = n02.to(device)
            out = model(im)
            # _, pred = torch.max(out.data, 1)
            loss = loss_function(out, n02)
            #print("Given loss", loss)
            losses.append(loss.item())

            #undo normalization
            real_out = (out * (max_n02 - min_n02)) + min_n02
            real_n02 = (n02 * (max_n02 - min_n02)) + min_n02
            real_mse = loss_function(real_out,real_n02)
            real_mae = np.abs(real_out - real_n02)
            #print("Got", out, "and wanted", n02, "with loss", loss)
            #print("Giving got", real_out, "and wanted", real_n02, "with real loss", real_mse)
            real_mses.append(real_mse)
            real_maes.append(real_mae)
            # y_true.extend(hr.tolist())
            # y_pred.extend(pred.tolist())
            # mses.append(mse(out, hr))
            # psnrs.append(psnr(out, hr))
            # ssims.append(ssim(out, hr))
    
    scores = {
        "real_mse":np.mean(real_mses),
        "real_mae":np.mean(real_maes)
    }
    valid_loss = np.mean(losses)
    return valid_loss, scores