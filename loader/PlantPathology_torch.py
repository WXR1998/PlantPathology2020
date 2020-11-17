from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
import torch

from .PlantPathology import *

class PlantPathology_torch(VisionDataset):
    plantPathology = PlantPathology()

    def __init__(
            self,
            root: str = None,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(PlantPathology_torch, self).__init__(root, transform=transform,
                                                   target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            raise NotImplementedError

        if self.train:
            x, y = self.plantPathology.trainData()
        else:
            x, y = self.plantPathology.trainData()
        self.data = torch.from_numpy(x)
        y = np.array(y, dtype=np.float32)
        self.targets = torch.from_numpy(y)
