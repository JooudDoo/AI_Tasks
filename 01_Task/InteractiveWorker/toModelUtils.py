
import numpy as np

import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F

CLASSES = np.array(['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble', 'bart_simpson',
    'carl_carlson', 'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler', 'comic_book_guy', 'disco_stu',
    'edna_krabappel', 'fat_tony', 'gil', 'groundskeeper_willie', 'homer_simpson', 'kent_brockman', 'krusty_the_clown', 'lenny_leonard',
    'lionel_hutz', 'lisa_simpson', 'maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby', 'milhouse_van_houten', 'miss_hoover',
    'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'otto_mann', 'patty_bouvier', 'principal_skinner', 'professor_john_frink', 'rainier_wolfcastle',
    'ralph_wiggum', 'selma_bouvier', 'sideshow_bob', 'sideshow_mel', 'snake_jailbird', 'troy_mcclure', 'waylon_smithers'])


IMAGE_RESIZE = (64, 64)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_DEVIATIONS = (0.5, 0.5, 0.5)

TRANSFORM = torchvision.transforms.Compose([
    # torchvision.transforms.ToPILImage(),
    transforms.Resize(IMAGE_RESIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_DEVIATIONS),
])

class LabelTransformer():
    def __init__(self, initial_list = CLASSES):
        self.initial_list = initial_list
    
    def __call__(self, val):
        if type(val) == str:
            return self.toInt(val)
        elif type(val) == int or type(val) == np.int64:
            return self.toStr(val)
        return None

    def toInt(self, label : str) -> int:
        return np.where(self.initial_list == label)[0][0]
    
    def toStr(self, ind : int) -> str:
        return self.initial_list[ind]

class myConvBlock(nn.Module):
    def __init__(self, inC : int, outC : int, kernel_size, **kwargs) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = inC
        self.out_channels = outC
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(inC, outC, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(outC)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, True)

class myInceptionBlock(nn.Module):
    def __init__(self, inC : int, outC : int, kernel_size : int) -> None:
        super().__init__()
        padding = int(((kernel_size-1)/2))
        self.cv1_1 = myConvBlock(inC, outC, 1)
        self.cv1_2 = myConvBlock(outC, outC, (1, kernel_size), padding=(0, padding), groups=outC)
        self.cv1_3 = myConvBlock(outC, outC, (kernel_size, 1), padding=(padding, 0), groups=outC)
    
    def forward(self, x):
        x = self.cv1_1(x)
        x = self.cv1_2(x)
        x = self.cv1_3(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            myConvBlock(3, 64, 3, padding=1),
            myConvBlock(64, 64, 2, stride=2, groups=64)
        )

        self.conv2 = nn.Sequential(
            myConvBlock(64, 128, 3, padding=1),
            nn.MaxPool2d(2)
        )

        self.residualBlock_1 = nn.Sequential(
            myConvBlock(128, 512, 1),
            myConvBlock(512, 512, kernel_size=3, groups=512, padding=1),
            myConvBlock(512, 128, 1),
        )

        self.conv3 = nn.Sequential(
            myConvBlock(128, 256, 3, padding=1),
            nn.MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
            myConvBlock(256, 512, 3, padding=1),
            nn.MaxPool2d(2)
        )

        self.residualBlock_2 = nn.Sequential(
            myConvBlock(512, 2048, 1),
            myConvBlock(2048, 2048, kernel_size=3, groups=2048, padding=1),
            myConvBlock(2048, 512, 1),
        )

        self.conv5 = nn.Sequential(
            myConvBlock(512, 1024, 3, padding=1),
            nn.MaxPool2d(2)
        )

        self.res = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.Linear(512, 128),
            nn.Linear(128, len(CLASSES)),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residualBlock_1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.residualBlock_2(x) + x
        x = self.conv5(x)
        x = self.res(x)
        return x
