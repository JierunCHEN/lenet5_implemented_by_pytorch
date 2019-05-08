import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    relu
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    relu
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    relu
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(
            OrderedDict(
                [
                    ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
                    ('relu1', nn.ReLU()),
                    ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
                    ('relu3', nn.ReLU()),
                    ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
                    ('relu5', nn.ReLU())
                ]
            )
        )

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ('f6', nn.Linear(120, 84)),
                    ('relu6', nn.ReLU()),
                    ('f7', nn.Linear(84, 10)),
                    ('sig7', nn.LogSoftmax(dim=-1))
                ]
            )
        )

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(-1, self.num_flat_features(x))
        output = self.fc(x)
        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
