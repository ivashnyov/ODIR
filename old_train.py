import torch
import os
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
import cv2
from torchvision import models

import collections
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    HorizontalFlip, VerticalFlip, CenterCrop, RandomRotate90, RandomCrop,
    PadIfNeeded, Normalize, Flip, OneOf, Compose, Resize, Transpose,
    IAAAdditiveGaussianNoise, GaussNoise, CLAHE, RandomBrightnessContrast, HueSaturationValue,
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)
from catalyst.dl import MetricCallback
from catalyst.contrib.schedulers import ReduceLROnPlateau, OneCycleLR
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, AUCCallback, F1ScoreCallback
from efficient_net.model import EfficientNet


def weighted_kappa(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = None,
        activation: str = None
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: quadratic kappa score
    """
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    outputs = outputs.flatten()
    targets = targets.flatten()
    outputs_clipped = outputs
    outputs_clipped[outputs_clipped < 0.5] = 0
    outputs_clipped[outputs_clipped >= 0.5] = 1
    score = cohen_kappa_score(outputs_clipped, targets)
    return score


class KappScoreMetricCallback(MetricCallback):
    """
    F1 score metric callback.
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "qkappa_score",
            activation: str = None
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """

        super().__init__(
            prefix=prefix,
            metric_fn=weighted_kappa,
            input_key=input_key,
            output_key=output_key,
            activation=activation
        )


def rocauc(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = None,
        activation: str = None
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: quadratic kappa score
    """
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    outputs = outputs.flatten()
    targets = targets.flatten()
    outputs_clipped = outputs
    outputs_clipped[outputs_clipped < 0.5] = 0
    outputs_clipped[outputs_clipped >= 0.5] = 1
    score = roc_auc_score(outputs, targets)
    return score


class RocAucCallback(MetricCallback):
    """
    F1 score metric callback.
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "rocauc_score",
            activation: str = None
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """

        super().__init__(
            prefix=prefix,
            metric_fn=rocauc,
            input_key=input_key,
            output_key=output_key,
            activation=activation
        )


def f1score(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = None,
        activation: str = None
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: quadratic kappa score
    """
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    outputs = outputs.flatten()
    targets = targets.flatten()
    outputs_clipped = outputs
    outputs_clipped[outputs_clipped < 0.5] = 0
    outputs_clipped[outputs_clipped >= 0.5] = 1
    score = f1_score(outputs_clipped, targets, average='micro')
    return score


class F1Callback(MetricCallback):
    """
    F1 score metric callback.
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "f1_score",
            activation: str = None
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """

        super().__init__(
            prefix=prefix,
            metric_fn=f1score,
            input_key=input_key,
            output_key=output_key,
            activation=activation
        )


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def preprocessing(image):
    image = crop_image_from_gray(image)
    return image


class EyesDataset(Dataset):
    def __init__(self, dataset_path, names, labels, augmentations):
        self.labels = labels
        self.names = names
        self.dataset_path = dataset_path
        self.augmentations = augmentations

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.dataset_path, self.names[index]))[..., :: -1]
        image = preprocessing(image)
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        target = self.labels[index]
        return torch.from_numpy(image.transpose((2, 0, 1))).float(), torch.tensor(target).float()


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

def aug_val(resolution, p=1):
    return Compose([Resize(resolution, resolution), Normalize()], p=p)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
    num_classes = 8
    lr = 3e-5
    IMG_SIZE = 256

    train = pd.read_csv('data/splited_train.csv')
    X = train['id'].values
    train = train.drop(columns='id')
    X_train, X_val, y_train, y_val = train_test_split(X, train.values.tolist(), test_size=0.2, random_state=42)

    model = EfficientNet.from_pretrained('efficientnet-b7')
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, 1)
    model.load_state_dict(torch.load('best_256.pth')['model_state_dict'])
    model._fc = nn.Linear(in_features, num_classes)
    model.cuda()

    train_dataset = EyesDataset(dataset_path='data/ODIR-5K_Training_Images',
                                labels=y_train,
                                names=X_train,
                                augmentations=aug_train_heavy(IMG_SIZE))

    val_dataset = EyesDataset(dataset_path='data/ODIR-5K_Training_Images',
                              labels=y_val,
                              names=X_val,
                              augmentations=aug_val(IMG_SIZE))

    train_loader = DataLoader(train_dataset,
                              num_workers=16,
                              pin_memory=False,
                              batch_size=32,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            num_workers=16,
                            pin_memory=False,
                            batch_size=32)

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = val_loader
    runner = SupervisedRunner()
    logdir = f"logs/baseline_b7_pretrained"
    for p in model.parameters():
        p.requires_grad = False
    for p in model._fc.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = 5
    criterion = nn.BCEWithLogitsLoss()
    scheduler = OneCycleLR(
        optimizer,
        num_steps=num_epochs,
        lr_range=(1e-4, 1e-7),
        init_lr=1e-5,
        warmup_fraction=0.5,
        momentum_range=(0.85, 0.98)
    )
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        scheduler=scheduler,
        callbacks=[
            KappScoreMetricCallback(),
            F1Callback(),
            EarlyStoppingCallback(patience=25, metric='loss')
        ],
        num_epochs=num_epochs,
        verbose=True
    )

    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = 15
    criterion = nn.BCEWithLogitsLoss()
    scheduler = OneCycleLR(
        optimizer,
        num_steps=num_epochs,
        lr_range=(1e-4, 1e-7),
        init_lr=1e-5,
        warmup_fraction=0.5,
        momentum_range=(0.85, 0.98)
    )
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        scheduler=scheduler,
        callbacks=[
            KappScoreMetricCallback(),
            F1Callback(),
            EarlyStoppingCallback(patience=25, metric='loss')
        ],
        num_epochs=num_epochs,
        verbose=True
    )