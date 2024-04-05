import torch.nn as nn
import torch.nn.functional as F

class Resblock(nn.Module):
    def __init__(self, inch, outch, ks=3, stride=1, padding=0, dilation=1, bias=False, groups=1):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(
            inch,
            outch,
            ks,
            stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
            ),
            nn.BatchNorm2d(outch),
            nn.ReLU(),
            nn.Conv2d(
                outch,
                outch,
                ks,
                stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                groups=groups,
            ),
            nn.BatchNorm2d(outch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.resblock(x)

class CustomResnet(nn.Module):
    def __init__(self, inch, num_classes=10):
        super().__init__()
        # prep layer
        self.prep_layer = nn.Sequential(
            nn.Conv2d(inch,64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # layer1
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer1_resblock = Resblock(128, 128, ks=3, stride=1, padding=1)


        # layer2
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # layer3
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer3_resblock = Resblock(512, 512, ks=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=4)
        self.fc = nn.Linear(512, num_classes, bias=False)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer1(x)
        x = x + self.layer1_resblock(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x + self.layer3_resblock(x)
        x = self.maxpool(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x #F.log_softmax(x, dim=-1)

