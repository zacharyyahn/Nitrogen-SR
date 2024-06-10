"""
Residual Channel Attention Network (RCAN) from https://arxiv.org/abs/1807.02758
"""
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torchsr.models import rcan
from metrics import mse, psnr, ssim


class net(nn.Module):
    def __init__(self, params):
        super(net, self).__init__()

        # Use a pretrained model
        self.model = rcan(int(params.scale), pretrained=True)

        #Freeze everything except for the upsampler
        i = 0
        for child in self.model.children():
            if i > 2 and i < 5:
                child.param.require_grad = False
            if i == 5:
                child.params.requires_grad = True
    
    def forward(self, im):

        #Make the input 3 channels instead of 1
        #out = torch.stack((im, im, im), axis=1)
        
        return self.model(im)
    
def train(model, device, train_loader, optimizer, loss_function):
    model.train()
    for (lr, hr) in tqdm(train_loader, leave=False):
        lr = lr.to(device)
        hr = hr.to(device)

        #Forward pass
        optimizer.zero_grad()
        out = model(lr)
        loss = loss_function(out, hr)

        #Backward pass
        loss.backward()
        optimizer.step()


def val(model, device, loader, loss_function):
    model.eval()
    y_true = []
    y_pred = []
    losses = []
    psnrs = []
    mses = []
    ssims = [] 
    with torch.no_grad():
        for lr, hr in tqdm(loader, leave=False):
            lr = lr.to(device)
            hr = hr.to(device)
            out = model(lr)
            _, pred = torch.max(out.data, 1)
            loss = loss_function(out, hr)
            losses.append(loss.item())
            y_true.extend(hr.tolist())
            y_pred.extend(pred.tolist())
            mses.append(mse(out, hr))
            psnrs.append(psnr(out, hr))
            ssims.append(ssim(out, hr))
    
    scores = {
        "psnr": np.mean(psnrs),
        "mse": np.mean(mses),
        "ssim": np.mean(ssims)
    }
    valid_loss = np.mean(losses)
    return valid_loss, scores