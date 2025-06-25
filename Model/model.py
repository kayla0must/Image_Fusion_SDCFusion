import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnLeakyRelu2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,
                              stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ConvBnTanh2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,
                              stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5


class ConvLeakyRelu2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,
                              stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class Conv1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,
                              stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)


from dehaze import DehazeNet, dehaze_image
from low_enhance import Low_enhance_net, low_enhance_image
from gate import GatedLayer


class fusion_module(nn.Module):

    def __init__(self, dim):
        super(fusion_module, self).__init__()
        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x, y):
        return self.conv(torch.cat([x, y], dim=1))

class F_Net(nn.Module):

    def __init__(self):
        super(F_Net, self).__init__()

        self.dehaze_net = DehazeNet()
        self.low_enhance_net = Low_enhance_net()
        self.gate_layer = GatedLayer()
        self.fusion = fusion_module(dim=128)


        self.conv1_vi = ConvBnLeakyRelu2d(1, 16)
        self.conv2_vi = ConvBnLeakyRelu2d(16, 32)
        self.conv3_vi = ConvBnLeakyRelu2d(32, 64)
        self.conv4_vi = ConvBnLeakyRelu2d(64, 128)


        self.conv1_ir = ConvBnLeakyRelu2d(1, 16)
        self.conv2_ir = ConvBnLeakyRelu2d(16, 32)
        self.conv3_ir = ConvBnLeakyRelu2d(32, 64)
        self.conv4_ir = ConvBnLeakyRelu2d(64, 128)


        self.decode1 = ConvBnLeakyRelu2d(128, 64)
        self.decode2 = ConvBnLeakyRelu2d(64, 32)
        self.decode3 = ConvBnLeakyRelu2d(32, 16)
        self.decode4 = ConvBnTanh2d(16, 1)

    def _enhance_stage(self, x, conv_layer, transmission, A, gate_factor):

        x = conv_layer(x)
        return dehaze_image(x, transmission, A) + x * (1 + gate_factor)

    def forward(self, vi, ir, feature):
        transmission, A = self.dehaze_net(vi)
        r = self.low_enhance_net(vi)
        gate_factor_1, gate_factor_2 = self.gate_layer(feature)

        x = self._enhance_stage(vi, self.conv1_vi, transmission, A, gate_factor_1)
        y = self.conv1_ir(ir)

        x = self._enhance_stage(x, self.conv2_vi, transmission, A, gate_factor_1)
        y = self.conv2_ir(y)

        x = self._enhance_stage(x, self.conv3_vi, transmission, A, gate_factor_1)
        y = self.conv3_ir(y)

        x = self._enhance_stage(x, self.conv4_vi, transmission, A, gate_factor_1)
        y = self.conv4_ir(y)

        f = self.fusion(x, y)

        f = self.decode1(f)
        f = self.decode2(f)
        f = self.decode3(f)
        f = self.decode4(f)

        return f


