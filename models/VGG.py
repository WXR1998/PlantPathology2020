import torch.nn as nn
import torch

from models.Model import Model

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(Model):
    epoch_num = 100

    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(384 * 512 // 32 // 32 * 512, self.class_num)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def get_features(self, x):
        '''
        :param x: Input tensor. shape = (n, 3, 384, 512)
        :return: Feature tensor. shape = (n, 98304)
        '''
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def get_classification(self, x):
        '''
        :param x: Input feature tensor. shape = (n, 98304)
        :return: Classification result. shape = (n, 4)
        '''
        return self.classifier(x)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG11(VGG):
    batch_size = 16
    def __init__(self):
        super(VGG11, self).__init__('VGG11')

class VGG13(VGG):
    batch_size = 10
    def __init__(self):
        super(VGG13, self).__init__('VGG13')

class VGG16(VGG):
    batch_size = 10
    def __init__(self):
        super(VGG16, self).__init__('VGG16')

class VGG19(VGG):
    batch_size = 8
    def __init__(self):
        super(VGG19, self).__init__('VGG19')
