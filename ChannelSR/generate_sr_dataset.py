from tqdm import tqdm
from torchvision import transforms
from utils.params import Params
import models.rcan_20m
import models.ninab1_20m
import models.ninab2_20m
import models.edsr
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

model_file = "ninab1_20m_v0"
save_path = "../Dataset/" + model_file + "_SR/"

model_dict = {
    "edsr": models.edsr,
    "rcan": models.rcan_20m,
    "ninab1": models.ninab1_20m,
    "ninab2": models.ninab2_20m
}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
trans = transforms.Compose([
    transforms.ToTensor()])

params = Params("saved_models/" + model_file + "_hparams.yaml", "DEFAULT")
model_type = model_file[:model_file.find("_")]
model_shell = model_dict[model_type]
trained_model = model_shell.net(params).to(device)
trained_model.load_state_dict(torch.load("saved_models/" + model_file + ".ckpt"))
trained_model.eval()

channels = [4, 5, 6, 7, 8, 9]
print("Generating dataset with model", model_file)
for path in tqdm(os.listdir("../Dataset/Unscaled/")):
    if path[path.find("_")+1:path.find(".")] in ["4", "5", "6", "7", "8", "9"]: 
        im = cv2.imread("../Dataset/Unscaled/" + path)
        im = im / 255.0
        input_im = trans(im).float()
        sr = 255.0 * trained_model(input_im).detach().numpy().reshape((3, 200, 200))[0, :, :]
        cv2.imwrite(save_path + path, sr)
    elif path[path.find("_")+1:path.find(".")] in ["0", "1", "2", "3"]:
        im = cv2.imread("../Dataset/Unscaled/" + path)
        cv2.imwrite(save_path + path, im)