"""
Nina models use a mixture of attention and residual blocks, from here: https://github.com/Coloquinte/torchSR/blob/main/doc/NinaSR.md
"""
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import math
from torchsr.models import ninasr_b1
from metrics import mse, psnr, ssim

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


class net(nn.Module):
    def __init__(self, params):
        n_resblocks=params.nina_res
        n_feats=params.nina_feats
        pretrained=False
        map_location=None
        expansion=2.0
        
        super(net, self).__init__()
        self.scale = int(params.scale)

        n_colors = 3
        self.head = net.make_head(n_colors, n_feats)
        self.body = net.make_body(n_resblocks, n_feats, expansion)
        self.tail = net.make_tail(n_colors, n_feats, self.scale)

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