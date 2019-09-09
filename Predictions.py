import sys
import torch 
import os
import random
import torch.nn as nn
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop
import time
from torch.autograd import Variable
import torch.functional as F
from tqdm import tqdm
from sklearn import metrics
import urllib
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error
import pickle
import cv2
import torch.nn.functional as F
from torchvision import models
import seaborn as sns
import random
import sys
import collections
import fire
from torch.utils.data import Dataset, WeightedRandomSampler, SubsetRandomSampler, DataLoader
from albumentations import (
    HorizontalFlip, VerticalFlip, CenterCrop, RandomRotate90, RandomCrop, 
    PadIfNeeded, Normalize, Flip, OneOf, Compose, Resize, Transpose, 
    IAAAdditiveGaussianNoise, GaussNoise, CLAHE, RandomBrightnessContrast, HueSaturationValue,
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)
from catalyst.contrib.schedulers import OneCycleLR, ReduceLROnPlateau, StepLR, MultiStepLR
from catalyst.dl.experiment import SupervisedExperiment
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, AccuracyCallback, F1ScoreCallback, ConfusionMatrixCallback, MixupCallback
from catalyst.dl.core.state import RunnerState
from catalyst.dl.core import MetricCallback
from catalyst.dl.callbacks import CriterionCallback
from efficientnet_pytorch import EfficientNet
from torch.nn.functional import softmax
from utils import *
from odir_submit import *
from pytorch_toolbelt.inference import tta
if __name__ == '__main__':
    splits = pickle.load(open('cv_split.pickle', 'rb'))
    labels  = ['N','D','G','C','A','H','M','O']
    n_classes = len(labels)
    fold_idx, batch_size, model_name, image_size, head_n_epochs, head_lr, full_n_epochs, full_lr, exp_name = fire.Fire(arguments)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    seed_everything(1234)
    test_data = pd.read_csv('./data/XYZ_ODIR.csv')
    test_data_left = test_data.copy()
    test_data_right = test_data.copy()
    test_data_left.loc[:,'id'] = test_data_left.ID.apply(lambda x: str(x)+'_left.jpg')
    test_data_right.loc[:,'id'] = test_data_left.ID.apply(lambda x: str(x)+'_right.jpg')
    test_data = pd.concat([test_data_left,test_data_right])
    test_data.sort_values(['ID'],inplace=True)
    test_path = './ODIR-5K_Testing_Images//'
    test_dataset = EyeDataset(dataset_path=test_path, 
                         labels=test_data.loc[:,labels].values, 
                         ids=test_data.loc[:,'id'].values, 
                         albumentations_tr=aug_val(image_size))
    test_loader =  DataLoader(test_dataset,
                         num_workers=8,
                         pin_memory=False,
                         batch_size=batch_size,
                         shuffle=False)  
    loaders = collections.OrderedDict()
    loaders["valid"] = test_loader    
    probabilities_list = []
    ttatype='d4'
    for fold_idx in range(len(splits['test_idx'])):
        print('Getting predictions from fold {}'.format(fold_idx))
        logdir = 'logs/{}_fold{}/'.format(exp_name, fold_idx)    
        model = prepare_model(model_name, n_classes)
        model.cuda()
        model.load_state_dict(torch.load(os.path.join(logdir,'checkpoints/best.pth'))['model_state_dict'])
        model.eval()   
        if ttatype=='d4':
            model = tta.TTAWrapper(model, tta.d4_image2label)
        elif ttatype=='fliplr_image2label':
            model = tta.TTAWrapper(model, tta.d4_image2label)
        runner = SupervisedRunner(model=model)   
        #predictions = runner.predict_loader(loaders["valid"], resume=f"{logdir}/checkpoints/best.pth")
        runner.infer(model=model,loaders=loaders,callbacks=[InferCallback()])
        predictions = runner.callbacks[0].predictions['logits']
        probabilities = softmax(torch.from_numpy(predictions),dim=1).numpy()    
        for idx in range(probabilities.shape[0]):
            if all(probabilities[idx,:]<0.5):
                probabilities[idx,0] = 1.0
        probabilities_list.append(probabilities)
    probabilities_combined = np.stack(probabilities_list,axis=0).mean(axis=0) 
    predicted_labels = pd.DataFrame(probabilities_combined, columns=labels)
    predicted_labels['id'] = test_data.loc[:,'id'].values
    predicted_labels.loc[:,'ID'] = predicted_labels.id.apply(lambda x: x.split('_')[0])
    predicted_labels_groupped = predicted_labels.groupby(['ID']).aggregate(dict(zip(labels,['max']*(len(labels)))))
    predicted_labels_groupped['ID'] = predicted_labels_groupped.index.values.astype(int)
    predicted_labels_groupped.reset_index(drop=True, inplace=True)
    predicted_labels_groupped.sort_values('ID',inplace=True)
    predicted_labels_groupped = predicted_labels_groupped.loc[:,['ID']+labels] 
    predicted_labels_groupped.to_csv('./submit_{}.csv'.format(exp_name),index=False)
