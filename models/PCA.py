import torch.nn as nn
import torch

from models.Model import Model

class PCA(Model):
    epoch_num = 30

    def __init__(self, in_dim, out_dim):
        super(PCA, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.classifier(x)
