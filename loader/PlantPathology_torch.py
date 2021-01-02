from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
import torch
import torch.utils.data as Data
from albumentations import *
from albumentations.pytorch import ToTensor

from .PlantPathology import *
from src import Logger as Log
plantPathology = None

class PlantPathology_torch(Data.Dataset):
    class_num = 4

    def __init__(self, subset='Train', valid_ratio=0.2, normalize=False):
        global plantPathology

        assert subset in ['Train', 'Test', 'Valid']
        assert valid_ratio >= 0 and valid_ratio <= 1

        if plantPathology is None:
            plantPathology = PlantPathology(use_cache=True)

        self.subset = subset
        if subset == 'Train' or subset == 'Valid':
            x, y = plantPathology.trainData()
            train_len = int(x.shape[0] * (1 - valid_ratio))
            if subset == 'Train':
                x = x[:train_len, ...]
                y = y[:train_len, ...]
            else:
                x = x[train_len:, ...]
                y = y[train_len:, ...]
        elif subset == 'Test':
            x, y = plantPathology.testData()
        else:
            raise NotImplementedError('Dataset subset error.')
        # x, y are numpy arrays here

        self.train_transform = Compose([HorizontalFlip(p=0.5),
                                        VerticalFlip(p=0.5),
                                        ShiftScaleRotate(rotate_limit=25, p=0.7),
                                        OneOf([IAAEmboss(p=1),
                                               IAASharpen(p=1),
                                               Blur(p=1)], p=0.5),
                                        IAAPiecewiseAffine(p=0.5)])
        self.test_transform = Compose([HorizontalFlip(p=0.5),
                                       VerticalFlip(p=0.5),
                                       ShiftScaleRotate(rotate_limit=25, p=0.7)])
        self.default_transform = Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), always_apply=True) if normalize is True else None,
                                          ToTensor()])  # normalized for pretrained network

        self.data = x
        self.targets = torch.from_numpy(y) if y.shape[1] > 0 else torch.zeros((y.shape[0], 4))

    @staticmethod
    def clear():
        global plantPathology
        plantPathology = None

    def __getitem__(self, index):
        assert index >= 0 and index < self.__len__()

        img, target = self.data[index], self.targets[index]
        if self.subset == 'Train':
            img = self.train_transform(image=img)['image']
        else:
            img = self.test_transform(image=img)['image']
        img = self.default_transform(image=img)['image']

        return img, target

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        return 'Dataset %s:\n\tShape = %s\n\tSize = %d' % (self.subset, self.data[0].shape, self.data.shape[0])
