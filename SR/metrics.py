import torch

def mse(im1, im2):
    sq_diff = torch.square(im1 - im2).reshape((3, 200, 200))
    mse = sq_diff.mean()
    return mse

def psnr(im1, im2):
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse(im1, im2)))
    return psnr

def ssim(im1, im2):
    k1 = 0.01
    k2 = 0.03
    L = 1.0
    c1 = L*k1 * L*k1
    c2 = L*k2 * L*k2
    mean_x = torch.mean(im1)
    mean_y = torch.mean(im2)
    var_x = torch.var(im1)
    var_y = torch.var(im2)
    cov_xy = torch.mean(torch.multiply(im1, im2)) - mean_x * mean_y
    ssim = ((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)) / ((mean_x * mean_x + mean_y * mean_y + c1) * (var_x + var_y + c2))
    return ssim

def dssim(im1, im2):
    return (1 - ssim(im1, im2)) / 2.0

def neg_psnr(im1, im2):
    return -1.0 * psnr(im1, im2)