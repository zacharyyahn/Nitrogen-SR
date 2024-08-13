import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet50
import math
import os

#Code borrowed extensively from https://github.com/Coloquinte/torchSR/blob/main/torchsr/models/ninasr.py
class AttentionBlock(nn.Module):
    """
    A typical Squeeze-Excite attention block, with a local pooling instead of global
    """

    def __init__(self, n_feats, reduction=4, stride=16):
        super(AttentionBlock, self).__init__()
        self.body = nn.Sequential(
            nn.AvgPool2d(
                2 * stride - 1,
                stride=stride,
                padding=stride - 1,
                count_include_pad=False,
            ),
            nn.Conv2d(n_feats, n_feats // reduction, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(n_feats // reduction, n_feats, 1, bias=True),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=stride, mode="nearest"),
        )

    def forward(self, x):
        res = self.body(x)
        if res.shape != x.shape:
            res = res[:, :, : x.shape[2], : x.shape[3]]
        return res * x


class ResBlock(nn.Module):
    def __init__(self, n_feats, mid_feats, in_scale, out_scale):
        super(ResBlock, self).__init__()

        self.in_scale = in_scale
        self.out_scale = out_scale

        m = []
        conv1 = nn.Conv2d(n_feats, mid_feats, 3, padding=1, bias=True)
        nn.init.kaiming_normal_(conv1.weight)
        nn.init.zeros_(conv1.bias)
        m.append(conv1)
        m.append(nn.ReLU(True))
        m.append(AttentionBlock(mid_feats))
        conv2 = nn.Conv2d(mid_feats, n_feats, 3, padding=1, bias=False)
        nn.init.kaiming_normal_(conv2.weight)
        # nn.init.zeros_(conv2.weight)
        m.append(conv2)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x * self.in_scale) * (2 * self.out_scale)
        res += x
        return res


class Rescale(nn.Module):
    def __init__(self, sign):
        super(Rescale, self).__init__()
        rgb_mean = (0.4488, 0.4371, 0.4040)
        bias = sign * torch.Tensor(rgb_mean).reshape(1, 3, 1, 1)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return x + self.bias


class nina_net(nn.Module):
    def __init__(self, params):
        n_resblocks=params.nina_res
        n_feats=params.nina_feats
        pretrained=False
        map_location=None
        expansion=2.0
        
        super(nina_net, self).__init__()
        self.scale = int(params.scale)

        n_colors = 3
        self.head = nina_net.make_head(n_colors, n_feats)
        self.body = nina_net.make_body(n_resblocks, n_feats, expansion)
        self.tail = nina_net.make_tail(n_colors, n_feats, self.scale)

        if pretrained:
            self.load_pretrained(map_location=map_location)

    @staticmethod
    def make_head(n_colors, n_feats):
        m_head = [
            Rescale(-1),
            nn.Conv2d(n_colors, n_feats, 3, padding=1, bias=False),
        ]
        return nn.Sequential(*m_head)

    @staticmethod
    def make_body(n_resblocks, n_feats, expansion):
        mid_feats = int(n_feats * expansion)
        out_scale = 4 / n_resblocks
        expected_variance = 1.0
        m_body = []
        for i in range(n_resblocks):
            in_scale = 1.0 / math.sqrt(expected_variance)
            m_body.append(ResBlock(n_feats, mid_feats, in_scale, out_scale))
            expected_variance += out_scale**2
        return nn.Sequential(*m_body)

    @staticmethod
    def make_tail(n_colors, n_feats, scale):
        m_tail = [
            nn.Conv2d(n_feats, n_colors * scale**2, 3, padding=1, bias=True),
            nn.PixelShuffle(scale),
            Rescale(1),
        ]
        return nn.Sequential(*m_tail)

    def forward(self, x, scale=None):
        if scale is not None and scale != self.scale:
            raise ValueError(f"Network scale is {self.scale}, not {scale}")
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

class net(nn.Module):
    def __init__(self, params):
        super(net, self).__init__()

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        #Initialize a custom nina architecture with 5 res layers and 16 features
        self.ninacustom = nina_net(params).to(device)

        #Create a ResNet50 architecture and change its conv1 layer to accept 12 channel inputs
        self.backbone = resnet50(weights="IMAGENET1K_V1")
        self.backbone.conv1 = torch.nn.Conv2d(params.num_inputs, 64, kernel_size=(3,3), stride=(2,2), padding=(3,3), bias=False)
        
        #Load in pretrained model from LUC dataset, which had 19 classes
        #self.backbone.load_state_dict(torch.load("./saved_models/pretrained_resnet50_LUC.model", map_location=device))

        #Reset the original fc layer
        self.backbone.fc = nn.Identity()

        #Add our own regression head
        self.head = nn.Sequential(nn.Linear(2048, 512), nn.Sigmoid(), nn.Linear(512, 1))

        #Try freezing everything but the head
        # for param in self.backbone.parameters():
        #     param.require_grad = True
        # for param in self.head.parameters():
        #     param.require_grad = True
    
    def forward(self, x):
        # We first need to SR on each image, then recombine them for the resnet
        #print("Taking in tensor of shape", x.shape)
        slices = torch.unbind(x, dim=1)
        #print("Got slices", len(slices), slices[0].shape)
        sr_outs = []
        for i, slice in enumerate(slices):
            slice = slice.squeeze()
            sr_in = torch.stack((slice, slice, slice))
            #print("Feeding SR", sr_in.shape)
            slice_out = self.ninacustom(sr_in)[0, 0, :, :].squeeze()
            #print("Have a slice of shape", slice_out.shape)
            sr_outs.append(slice_out)
        sr_out = torch.stack(sr_outs).unsqueeze(0)
        #print("Feeding resnet shape", sr_out.shape)
        x = self.backbone(x)
        x = self.head(x)
        return x

def train(model, device, train_loader, optimizer, loss_function):
    model.train()
    # print("--- TRAIN ---")
    for (im, n02) in tqdm(train_loader, leave=False):
        im = im.to(device)
        n02 = n02.to(device)
        #print(f"Im has max {im.max()} and min {im.min()}")
        #print("Have n02 value", n02)

        #Forward pass
        optimizer.zero_grad()
        out = model(im)
        loss = loss_function(out, n02)
        # print("Wanted", n02, "got", out)
        # print("Got train loss", loss.item())
        #print("Got loss", loss)

        #print("Expected", n02, "got", out, "propagating loss", loss)

        #Backward pass
        loss.backward()
        optimizer.step()

def val(model, device, loader, loss_function):
    # print("--- VAL ---")
    max_n02 = 71.75532137
    min_n02 = -2.68378695
    avg_n02 = 18.71056311
    std_n02 = 13.1422108
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
            # print("Wanted:", n02, "got", out)
            # print("loss", loss.item())
            #print("Given loss", loss)
            losses.append(loss.item())

            #undo normalization
            # real_out = (out * (max_n02 - min_n02)) + min_n02
            # real_n02 = (n02 * (max_n02 - min_n02)) + min_n02
            real_out = (out * std_n02 + avg_n02)
            real_n02 = (n02 * std_n02 + avg_n02)
            real_mse = np.abs(real_out - real_n02) * np.abs(real_out - real_n02)
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