import torch
import torch.nn as nn
import os

from src import Logger as Log

class Model(nn.Module):
    model_name = None

    batch_size = 16
    epoch_num = 50
    save_interval = 10
    class_num = 4

    def __init__(self):
        super(Model, self).__init__()
        self.model_name = self.__class__.__name__

    def forward(self, x):
        raise NotImplementedError

    def save(self, path: str):
        root_dir = './ckpt/%s/' % self.model_name
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        if path is None or not path.endswith('.pth'):
            Log.log(Log.ERROR, 'Save filename should end with .pth.')
            raise ValueError

        pth_path = os.path.join(root_dir, path)
        torch.save(self.state_dict(), pth_path)
        Log.log(Log.INFO, 'Model is saved at [%s].' % pth_path)

    def load(self, path=None):
        '''
        :param path: The filename of the .pth file. '.pth' is optional.
        :return: int. Indicate the epoch number of the pth file.
        '''
        root_dir = './ckpt/%s/' % self.model_name

        pth_path = None
        if path is None:
            lists = os.listdir(root_dir)
            lists.sort(key=lambda x: os.path.getmtime(os.path.join(root_dir, x)))
            for i in range(len(lists)):
                if lists[len(lists) - i - 1].endswith('.pth'):
                    pth_path = lists[len(lists) - i - 1]
                    break
        else:
            pth_path = path

        if pth_path is not None and not pth_path.endswith('.pth'):
            pth_path = pth_path + '.pth'

        if pth_path is None or not os.path.exists(os.path.join(root_dir, pth_path)):
            Log.log(Log.ERROR, 'No checkpoint file found or invalid checkpoint filename.')
            raise RuntimeError

        pth_path = os.path.join(root_dir, pth_path)
        state_dict = torch.load(pth_path)
        self.load_state_dict(state_dict)
        Log.log(Log.INFO, 'Model is loaded from [%s].' % os.path.join(pth_path))

        return int(pth_path.split('/')[-1][:-4])
