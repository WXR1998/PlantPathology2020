__all__ = ['AlexNet', 'LeNet', 'Model', 'TestModel', 'VGG', 'ResNet', 'PCA', 'EfficientNet']
from . import *

_model_dict = {
    'AlexNet': AlexNet.AlexNet,
    'LeNet': LeNet.LeNet,
    'TestModel': TestModel.TestModel,
    'VGG11': VGG.VGG11,
    'VGG13': VGG.VGG13,
    'VGG16': VGG.VGG16,
    'VGG19': VGG.VGG19,
    'ResNet18': ResNet.ResNet18,
    'ResNet34': ResNet.ResNet34,
    # 'ResNet50': ResNet.ResNet50,
    # 'ResNet101': ResNet.ResNet101,
    # 'ResNet152': ResNet.ResNet152,
    'PCA': PCA.PCA,
    'EfficientNet': EfficientNet.EfficientNet,
}

def import_model(st):
    if st in _model_dict.keys():
        return _model_dict[st]
    else:
        raise KeyError('Model name error.')