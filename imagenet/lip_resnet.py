import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet50': './lip_resnet-50.pth',
    'resnet101': './lip_resnet-101.pth',
}

BOTTLENECK_WIDTH = 128
COEFF = 12.0

def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x*weight, kernel, stride, padding)/F.avg_pool2d(weight, kernel, stride, padding)

class SoftGate(nn.Module):
    def __init__(self):
        super(SoftGate, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x).mul(COEFF)


class BottleneckShared(nn.Module):
    def __init__(self, channels):
        super(BottleneckShared, self).__init__()
        rp = BOTTLENECK_WIDTH

        self.logit = nn.Sequential(
            OrderedDict((
                ('conv1', conv1x1(channels, rp)),
                ('bn1', nn.InstanceNorm2d(rp, affine=True)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', conv3x3(rp, rp)),
                ('bn2', nn.InstanceNorm2d(rp, affine=True)),
                ('relu2', nn.ReLU(inplace=True)),
            ))
        )

    def init_layer(self):
        pass

    def forward(self, x):
        return self.logit(x)

class BottleneckLIP(nn.Module):
    def __init__(self, channels):
        super(BottleneckLIP, self).__init__()
        rp = BOTTLENECK_WIDTH

        self.postprocessing = nn.Sequential(
            OrderedDict((
                ('conv', conv1x1(rp, channels)),
                ('bn', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate()),
            ))
        )

    def init_layer(self):
        self.postprocessing[0].weight.data.fill_(0.0)
        pass

    def forward_with_shared(self, x, shared):
        frac = lip2d(x, self.postprocessing(shared))
        return frac


class SimplifiedLIP(nn.Module):
    def __init__(self, channels):
        super(SimplifiedLIP, self).__init__()

        rp = channels

        self.logit = nn.Sequential(
            OrderedDict((
                ('conv', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
                ('bn', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate()),
            ))
        )

    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x))
        return frac


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def init_layer(self):
#         self.bn2.weight.data.zero_()

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        if stride == 2:
            kplanes = planes

            self.bottleneck_shared = BottleneckShared(inplanes)

            self.conv1 = conv1x1(inplanes, planes)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Sequential(
                BottleneckLIP(planes),
                conv1x1(planes, planes),
            )
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = conv1x1(planes, planes * self.expansion)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        else:
            self.conv1 = conv1x1(inplanes, planes)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = conv1x1(planes, planes * self.expansion)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def init_layer(self):
        self.bn3.weight.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.stride == 2:
            shared = self.bottleneck_shared(x)

            out = self.conv2[0].forward_with_shared(out, shared)
            for layer in self.conv2[1:]:
                out = layer(out)
        else:
            out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            if self.stride == 1:
                residual = self.downsample(x)
            else:
                residual = self.downsample[0].forward_with_shared(x, shared)
                for layer in self.downsample[1:]:
                    residual = layer(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = SimplifiedLIP(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()

        for m in self.modules():
            if hasattr(m, 'init_layer'):
                m.init_layer()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None


        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride != 1:
                downsample = OrderedDict([
                    ('lip', BottleneckLIP(self.inplanes)),
                    ('conv', conv1x1(self.inplanes, planes * block.expansion)),
                    ('bn', nn.BatchNorm2d(planes * block.expansion)),
                ])
            else:
                downsample = OrderedDict([
                    ('conv', conv1x1(self.inplanes, planes * block.expansion, stride)),
                    ('bn', nn.BatchNorm2d(planes * block.expansion)),
                ])

        downsample = nn.Sequential(downsample)
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


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model.load_state_dict(torch.load(model_urls['resnet50'], map_location='cpu'))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        model.load_state_dict(torch.load(model_urls['resnet101'], map_location='cpu'))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


if __name__ == '__main__':

    model = resnet50()
    print(model)
    model(torch.randn((1, 3, 224, 224)))
