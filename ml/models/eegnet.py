import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, num_classes, num_channels=19, input_length=500):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32))
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (num_channels, 1))
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * (input_length // 2), num_classes)

    def forward(self, x):
        x = F.elu(self.batchnorm1(self.conv1(x)))
        x = F.elu(self.batchnorm2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
