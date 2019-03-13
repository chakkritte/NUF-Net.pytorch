import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from torch.autograd import Variable

__all__ = ['NUFNet', 'nufnet8', 'nufnet16','nufnet33','nufnet50']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class NUFBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, in_planes , stride=1, downsample=None):
        super(NUFBlock, self).__init__()
        
        output_reduce = int(in_planes/2)
        
        self.reduce = nn.Conv2d(inplanes, output_reduce, kernel_size=1, stride=1, padding=0)
        
        output = int(in_planes/4)
        
        # 1x1 conv
        self.b1 = nn.Sequential(
            conv1x1(output_reduce, output, stride=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
        )
        
        # 3x3 conv
        self.b2 = nn.Sequential(
            nn.Conv2d(output_reduce, output, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
        )   
        
        # 3x3 double conv
        self.b3 = nn.Sequential(
            conv3x3(output_reduce, output, stride=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
            
            conv3x3(output, output, stride=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
        )
        
        # 3x3 triple conv
        self.b4 = nn.Sequential(
            conv3x3(output_reduce, output, stride=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),

            conv3x3(output, output, stride=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
            
            conv3x3(output, output, stride=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
        )            

        
        self.conv2d = nn.Conv2d(output*4, in_planes, kernel_size=1, stride=stride, padding=0)
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        
        
    def forward(self, x):
        
        residual = x
        
        x = self.reduce(x)
        
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        out = torch.cat([y1,y2,y3,y4], 1)
        
        out = self.conv2d(out)
        out = self.bn(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        
        out = self.relu(out)
        
        return out


class NUFNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(NUFNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def nufnet8(num_classes=1000):
    model = NUFNet(NUFBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def nufnet16(num_classes=1000):
    model = NUFNet(NUFBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model

def nufnet33(num_classes=1000):
    model = NUFNet(NUFBlock, [3, 4, 23, 3], num_classes=num_classes)
    return model

def nufnet50(num_classes=1000):
    model = NUFNet(NUFBlock, [3, 8, 36, 3], num_classes=num_classes)
    return model

nufnet = {
    'nufnet8' : nufnet8,
    'nufnet16': nufnet16,
    'nufnet33': nufnet33,
    'nufnet50': nufnet50,
}

if __name__ == '__main__':
    model = nufnet50(1000)
    x_image = Variable(torch.randn(1, 3, 224, 224))
    y = model(x_image)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total params: ",pytorch_total_params)
    
    print(y.size())
