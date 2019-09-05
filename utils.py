import numpy as np
import cv2
import torch
import torch.nn as nn
import random
import os
from albumentations import (
    HorizontalFlip, VerticalFlip, CenterCrop, RandomRotate90, RandomCrop, 
    PadIfNeeded, Normalize, Flip, OneOf, Compose, Resize, Transpose, 
    IAAAdditiveGaussianNoise, GaussNoise, CLAHE, RandomBrightnessContrast, HueSaturationValue,
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)
from torch.utils.data import Dataset
from efficientnet_pytorch import EfficientNet
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def arguments(fold_idx, batch_size, model_name, image_size, head_n_epochs, head_lr, full_n_epochs, full_lr, exp_name):
	print('Will run with the following arguments: \n fold_idx : {}, model_name : {}, batch_size : {} \n image_size : {}, head_n_epochs : {}, head_lr : {}, full_n_epochs : {}, full_lr : {}, exp_name : {}'.format(fold_idx, model_name, batch_size, image_size, head_n_epochs, head_lr, full_n_epochs, full_lr, exp_name))
	return(fold_idx, batch_size, model_name, image_size, head_n_epochs, head_lr, full_n_epochs, full_lr, exp_name)

def prepare_model(model_name, n_classes):
	print('Will load {}'.format(model_name))
	model = EfficientNet.from_pretrained(model_name)
	in_features = model._fc.in_features
	model._fc = nn.Linear(in_features, n_classes)
	model.cuda()
	return(model)		

def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True

class EyeDataset(Dataset):
    def __init__(self, dataset_path, labels, ids, albumentations_tr=False, shuffle=True):
        self.labels = labels
        self.ids = ids
        self.shuffle = shuffle
        self.dataset_path = dataset_path
        self.albumentations_tr = albumentations_tr
            
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        #imid = self.ids[index]
        image = cv2.imread(os.path.join(self.dataset_path, self.ids[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        if self.albumentations_tr:
            augmented = self.albumentations_tr(image=image)
            image = augmented['image']
        target = np.argmax(self.labels[index])
        #return torch.from_numpy(image.transpose((2, 0, 1))).float(), torch.tensor(np.expand_dims(target,0)).long()
        return torch.from_numpy(image.transpose((2, 0, 1))).float(), torch.tensor(target).long()

def aug_train(resolution,p=1): 
    return Compose([Resize(resolution,resolution),
                    OneOf([
                        HorizontalFlip(), 
                        VerticalFlip(), 
                        RandomRotate90(), 
                        Transpose()],p=0.5),
                    Normalize()
                    ], p=p)
def aug_train_heavy(resolution, p=1.0):
    return Compose([
        Resize(resolution,resolution),
        OneOf([RandomRotate90(),Flip(),Transpose(),HorizontalFlip(),VerticalFlip()],p=1.0),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.5),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.1),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),            
        ], p=0.3),
        HueSaturationValue(p=0.3),
        Normalize()
    ], p=p)

def aug_val(resolution,p=1): 
    return Compose([Resize(resolution,resolution),Normalize()], p=p)
