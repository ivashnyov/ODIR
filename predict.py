import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from train import preprocessing, aug_val
from tqdm import tqdm
import cv2


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_classes = 8
    IMG_SIZE = 256

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, num_classes)
    model.load_state_dict(torch.load('logs/baseline_crop_from_gray/checkpoints/best_full.pth')['model_state_dict'])
    model.cuda()
    model.eval()

    test = pd.read_csv('data/XYZ_ODIR.csv')
    with torch.no_grad():
        for index, row in tqdm(test.iterrows()):
            image_left = cv2.imread(os.path.join('data/ODIR-5K_Testing_Images', str(row['ID']) + '_left.jpg'))[..., :: -1]
            image_left = preprocessing(image_left)
            augmented = aug_val(IMG_SIZE)(image=image_left)
            image_left = augmented['image']
            image_left = np.expand_dims(image_left, 0)
            out_left = model(torch.from_numpy(image_left.transpose((0, 3, 1, 2))).float().cuda())
            predict_left = nn.Sigmoid()(out_left)
            predict_left[predict_left < 0.5] = 0
            predict_left[predict_left >= 0.5] = 1

            image_right = cv2.imread(os.path.join('data/ODIR-5K_Testing_Images', str(row['ID']) + '_right.jpg'))[..., :: -1]
            image_right = preprocessing(image_right)
            augmented = aug_val(IMG_SIZE)(image=image_right)
            image_right = augmented['image']
            image_right = np.expand_dims(image_right, 0)
            out_right = model(torch.from_numpy(image_right.transpose((0, 3, 1, 2))).float().cuda())
            predict_right = nn.Sigmoid()(out_right)
            predict_left[predict_right < 0.5] = 0
            predict_left[predict_right >= 0.5] = 1

            

            print(nn.Sigmoid()(out_left), nn.Sigmoid()(out_right))