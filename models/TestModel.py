import torch.nn as nn

from models.Model import Model

class TestModel(Model):
    model_name = 'TestModel'

    epoch_num = 50

    def __init__(self):
        super(TestModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(nn.Linear(24 * 32 * 16, 1024),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(1024, 4))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 24 * 32 * 16)
        x = self.dense(x)
        return x
