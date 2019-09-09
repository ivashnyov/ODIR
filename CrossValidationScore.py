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
if __name__=='__main__':
    splits = pickle.load(open('cv_split.pickle', 'rb'))
    data = pd.read_csv('./data/splited_train.csv')
    labels  = ['N','D','G','C','A','H','M','O']
    n_classes = len(labels)
    fold_idx, batch_size, model_name, image_size, head_n_epochs, head_lr, full_n_epochs, full_lr, exp_name = fire.Fire(arguments)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_classes = len(labels)
    seed_everything(1234)
    train_path = './ODIR-5K_Training_Dataset/'
    valid_path = './ODIR-5K_Training_Dataset/'
    model = prepare_model(model_name, n_classes)
    model.cuda()
    model.eval()
    for fold_idx in range(len(splits['test_idx'])):
        valid_data_groupped, predicted_labels_groupped = run_validation(data, valid_path, image_size, batch_size, splits, fold_idx, model, exp_name, labels)
        kappa, f1, auc, final_score = ODIR_Metrics(valid_data_groupped.loc[:,labels].values, 
                                                   predicted_labels_groupped.loc[:,labels].values)
        print("Fold ", fold_idx, " kappa score:", kappa, " f-1 score:", f1, " AUC vlaue:", auc, " Final Score:", final_score)        
