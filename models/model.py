import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, num_class):
        super(Model, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 16, (3, 3)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(16, 32, (3, 3)),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(32))
        # input 3*30*30 output 32*13*13

        self.block2 = nn.Sequential(nn.Conv2d(32, 64, (3, 3)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 128, (3, 3)),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(128))
        # input 32*13*13 output 128*4*4

        self.block3 = nn.Sequential(nn.Flatten(),
                                    nn.Linear(128*4*4, 512),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm1d(512),
                                    nn.Dropout(p=0.5))

        self.block4 = nn.Sequential(nn.Linear(512, num_class),
                                    nn.Softmax())

    def forward(self, x):
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.block3(x)
        # print(x.shape)
        x = self.block4(x)
        # print(x.shape)
        return x


if __name__ == '__main__':
    import numpy as np

    model = Model(43).cuda()
    x = torch.autograd.Variable(torch.rand(3, 3, 30, 30)).cuda()
    print(x.shape)
    y = model(x)
    print(y)
