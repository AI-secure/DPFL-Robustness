import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet

class CifarNet(SimpleNet):
    def __init__(self, name=None, created_time=None):
        super(CifarNet, self).__init__(f'{name}_CifarNet', created_time)

        self.net = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, 10, bias=True),
    )

    def forward(self, x):
        x = self.net(x)
        
        return x



class CifarNet100(SimpleNet):
    def __init__(self, name=None, created_time=None):
        super(CifarNet100, self).__init__(f'{name}_CifarNet100', created_time)

        self.net = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        
    )
        self.fc= nn.Linear(128, 100, bias=True)

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        
        return x


    