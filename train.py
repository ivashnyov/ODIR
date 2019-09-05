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
from utils import *
if __name__ == '__main__':
	splits = pickle.load(open('cv_split.pickle', 'rb'))
	data = pd.read_csv('./data/splited_train.csv')
	labels  = ['N','D','G','C','A','H','M','O']
	n_classes = len(labels)
	fold_idx, batch_size, model_name, image_size, head_n_epochs, head_lr, full_n_epochs, full_lr, exp_name = fire.Fire(arguments)
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	num_classes = len(labels)
	seed_everything(1234)
	runner = SupervisedRunner()
	model = prepare_model(model_name, n_classes)	
	train_path = './ODIR-5K_Training_Dataset/'
	valid_path = './ODIR-5K_Training_Dataset/'
	train_dataset = EyeDataset(dataset_path = train_path, 
					labels=data.loc[splits['train_idx'][fold_idx],labels].values, 
					ids=data.loc[splits['train_idx'][fold_idx],'id'].values, 
					albumentations_tr=aug_train_heavy(image_size)) 
	val_dataset = EyeDataset(dataset_path=valid_path, 
					labels=data.loc[splits['train_idx'][fold_idx],labels].values, 
					ids=data.loc[splits['train_idx'][fold_idx],'id'].values, 
					albumentations_tr=aug_val(image_size))
	train_loader =  DataLoader(train_dataset,
					num_workers=4,
					pin_memory=False,
					batch_size=batch_size,
					shuffle=True)
	val_loader =  DataLoader(val_dataset,
					num_workers=4,
					pin_memory=False,
					batch_size=batch_size,
					shuffle=True)	
	loaders = collections.OrderedDict()
	loaders["train"] = train_loader
	loaders["valid"] = val_loader
	logdir = 'logs/{}_fold{}/'.format(exp_name, fold_idx)
	print('Training only head for {} epochs with inital lr {}'.format(head_n_epochs, head_lr))
	for p in model.parameters():
		p.requires_grad = False
	for p in model._fc.parameters():
		p.requires_grad = True
	optimizer = torch.optim.Adam(model.parameters(), lr=head_lr)
	criterion = nn.CrossEntropyLoss()
	scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=2)
	runner.train(model=model,
			criterion=criterion,
			optimizer=optimizer,
			loaders=loaders,
			logdir=logdir,
			scheduler=scheduler,
			callbacks=[
				],
			num_epochs=head_n_epochs,
			verbose=True)      
	print('Train whole net for {} epochs with initial lr {}'.format(full_n_epochs, full_lr))
	for p in model.parameters():
		p.requires_grad = True
	optimizer = torch.optim.Adam(model.parameters(), lr=full_lr)
	criterion = nn.CrossEntropyLoss()
	scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=2)
	runner.train(model=model,
			criterion=criterion,
			optimizer=optimizer,
			loaders=loaders,
			logdir=logdir,
			scheduler=scheduler,
			callbacks=[
				],
			num_epochs=full_n_epochs,
			verbose=True)    	
