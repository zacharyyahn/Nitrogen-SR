from tqdm import tqdm
from torchvision import transforms
from utils.params import Params
import models.rcan
import models.ninab1
import models.ninab2
import models.edsr
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

model_file = "ninab1_v6"
save_path = "../Dataset/" + model_file + "_SR/"

model_dict = {
    "edsr": models.edsr,
    "rcan": models.rcan,
    "ninab1": models.ninab1,
    "ninab2": models.ninab2
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

print("Generating dataset with model", model_file)
for path in tqdm(os.listdir("../Dataset/Formatted/")):
    im = cv2.imread("../Dataset/Formatted/" + path)
    im = im[:, :, 0]
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    input_im = trans(im).float()
    sr = 255.0 * trained_model(input_im).detach().numpy().reshape((3, 400, 400))[0, :, :]
    cv2.imwrite(save_path + path, sr)