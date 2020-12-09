from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
import torch

from .PlantPathology import *
from src import Logger as Log
plantPathology = None

class PlantPathology_torch(VisionDataset):
    class_num = 4

    def __init__(
        self,
        root: str = None,
        subset: str = 'Train',
        valid_ratio: float = 0.1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        global plantPathology
        super(PlantPathology_torch, self).__init__(root, transform=transform,
                                                   target_transform=target_transform)
        assert subset in ['Train', 'Test', 'Valid']
        assert valid_ratio >= 0 and valid_ratio <= 1
        if download:
            raise NotImplementedError

        if plantPathology is None:
            plantPathology = PlantPathology(use_cache=True)

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

        Log.log(Log.INFO, 'Converting to tensor...')
        self.data = torch.from_numpy(x)
        y = np.array(y, dtype=np.float16)
        self.targets = torch.from_numpy(y)
        self.mean = plantPathology.mean
        self.subset = subset

    @staticmethod
    def clear():
        global plantPathology
        plantPathology = None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        assert index >= 0 and index < self.__len__()

        img, target = self.data[index], self.targets[index]

        img = img.numpy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        return 'Dataset %s:\n\tShape = %s\n\tSize = %d' % (self.subset, self.data[0].shape, self.data.shape[0])
