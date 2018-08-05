import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import torchvision
import torch.nn as nn
import math
from cropextract import *
import scipy.io as sio
import scipy.misc as smi

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetMI(nn.Module):
    def __init__(self, block, layers, num_classes=256):
        super(ResNetMI, self).__init__()
        self.inplanes = 64
        self.inplanes_rgbdnm = 4
        super(ResNetMI, self).__init__()

        self.conv1 = nn.Conv2d(3, 4, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(1, 4, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(3, 4, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm2d(4)
        self.conv4 = nn.Conv2d(1, 4, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn4 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      
        self.layer1_color = self._make_layer_rgbdnm(block, 4, layers[0])
        self.layer1_depth = self._make_layer_rgbdnm(block, 4, layers[0])
        self.layer1_normal = self._make_layer_rgbdnm(block, 4, layers[0])
        self.layer1_mask = self._make_layer_rgbdnm(block, 4, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.fcms = nn.Linear(1 * num_classes, num_classes)
        self.fcms_gl = nn.Linear(2 * num_classes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_rgbdnm(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_rgbdnm != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_rgbdnm, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_rgbdnm, planes, stride, downsample))
        self.inplanes_rgbdnm = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_rgbdnm, planes))

        self.inplanes_rgbdnm = 4
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        x2 = x2[:,None, :, :]
        x4 = x4[:,None, :, :]
        x6 = x6[:,None, :, :]
        x8 = x8[:,None, :, :]
        
        # global tower
        # conv1-11
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.layer1_color(x1)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.layer1_depth(x2)
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x3 = self.maxpool(x3)
        x3 = self.layer1_normal(x3)
        x4 = self.conv4(x4)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        x4 = self.maxpool(x4)
        x4 = self.layer1_mask(x4)
        x1 = torch.cat((x1, x2, x3, x4), 1)

        # conv12-50
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc(x1)
        xms1 = self.fcms(x1)

        # local tower
        # conv1-11
        x5 = self.conv1(x5)
        x5 = self.bn1(x5)
        x5 = self.relu(x5)
        x5 = self.maxpool(x5)
        x5 = self.layer1_color(x5)
        x6 = self.conv2(x6)
        x6 = self.bn2(x6)
        x6 = self.relu(x6)
        x6 = self.maxpool(x6)
        x6 = self.layer1_depth(x6)
        x7 = self.conv3(x7)
        x7 = self.bn3(x7)
        x7 = self.relu(x7)
        x7 = self.maxpool(x7)
        x7 = self.layer1_normal(x7)
        x8 = self.conv4(x8)
        x8 = self.bn4(x8)
        x8 = self.relu(x8)
        x8 = self.maxpool(x8)
        x8 = self.layer1_mask(x8)
        x5 = torch.cat((x5, x6, x7, x8), 1)

        # conv12-50
        x5 = self.layer2(x5)
        x5 = self.layer3(x5)
        x5 = self.layer4(x5)
        x5 = self.avgpool(x5)
        x5 = x5.view(x5.size(0), -1)
        x5 = self.fc(x5)
        xms2 = self.fcms(x5)

        # concat global & local
        xms = torch.cat((xms1, xms2), 1)
        xms = self.fcms_gl(xms)
        return xms
