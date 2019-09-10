import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from old_train import preprocessing, aug_val
from tqdm import tqdm
import cv2
from efficient_net.model import EfficientNet

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_classes = 8
    IMG_SIZE = 256

    model = EfficientNet.from_pretrained('efficientnet-b7')
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load('logs/baseline_b7_pretrained/checkpoints/best_full.pth')['model_state_dict'])
    model.cuda()
    model.eval()

    test = pd.read_csv('data/XYZ_ODIR.csv')
    answers = []
    with torch.no_grad():
        for index, row in tqdm(test.iterrows()):
            image_left = cv2.imread(os.path.join('data/ODIR-5K_Testing_Images', str(row['ID']) + '_left.jpg'))[..., :: -1]
            image_left = preprocessing(image_left)
            augmented = aug_val(IMG_SIZE)(image=image_left)
            image_left = augmented['image']
            image_left = np.expand_dims(image_left, 0)
            out_left = model(torch.from_numpy(image_left.transpose((0, 3, 1, 2))).float().cuda())
            predict_left = nn.Sigmoid()(out_left).cpu().numpy()[0]
            if not (predict_left < 0.5).all():
                predict_left[predict_left < 0.5] = 0
                predict_left[predict_left >= 0.5] = 1
            else:
                predict_left[np.where(predict_left == np.max(predict_left))] = 1
                predict_left[predict_left < 1] = 0


            image_right = cv2.imread(os.path.join('data/ODIR-5K_Testing_Images', str(row['ID']) + '_right.jpg'))[..., :: -1]
            image_right = preprocessing(image_right)
            augmented = aug_val(IMG_SIZE)(image=image_right)
            image_right = augmented['image']
            image_right = np.expand_dims(image_right, 0)
            out_right = model(torch.from_numpy(image_right.transpose((0, 3, 1, 2))).float().cuda())
            predict_right = nn.Sigmoid()(out_right).cpu().numpy()[0]
            if not (predict_right < 0.5).all():
                predict_right[predict_right < 0.5] = 0
                predict_right[predict_right >= 0.5] = 1
            else:
                predict_right[np.where(predict_right == np.max(predict_right))] = 1
                predict_right[predict_right < 1] = 0
            # print('pr_left ', predict_left, ' pr_right ', predict_right)
            if predict_left[0] == 1 and predict_right[0] == 1:
                ans = [0] * 9
                ans[0] = row['ID']
                ans[1] = 1
                answers.append(ans)
            else:
                ans = [0] * 9
                ans[0] = row['ID']
                for i in range(len(predict_left)):
                    if predict_left[i] == 1 or predict_right[i] == 1:
                        ans[i + 1] = 1
                ans[1] = 0
                answers.append(ans)

        final_df = pd.DataFrame(columns=test.columns, data=answers)
        final_df.to_csv('answers.csv', index=False)

            # print(nn.Sigmoid()(out_left), nn.Sigmoid()(out_right))