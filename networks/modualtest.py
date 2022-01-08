from legomodel.attention.AFT import AFT_FULL
import torch
from torch import nn
from torch.nn import functional as F

import SimpleITK as sitk
'''
class SpixelNet(nn.Module):
    expansion = 1
    def __init__(self, input_channel=3, batchNorm=True):
        super(SpixelNet, self).__init__()

        self.batchNorm = batchNorm
        self.assign_ch = 9

        self.conv0a = conv(self.batchNorm, input_channel, input_channel * 2, kernel_size=3)
        self.conv0b = conv(self.batchNorm, input_channel * 2, input_channel * 2, kernel_size=3)

        self.conv1a = conv(self.batchNorm, input_channel * 2, input_channel * 4, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, input_channel * 4, input_channel * 4, kernel_size=3)

        self.conv2a = conv(self.batchNorm, input_channel * 4, input_channel * 8, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, input_channel * 8, input_channel * 8, kernel_size=3)

        self.conv3a = conv(self.batchNorm, input_channel * 8, input_channel * 16, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, input_channel * 16, input_channel * 16, kernel_size=3)

        self.conv4a = conv(self.batchNorm, input_channel * 16, input_channel * 32, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, input_channel * 32, input_channel * 32, kernel_size=3)

        self.deconv3 = deconv(input_channel * 32, input_channel * 16)
        self.conv3_1 = conv(self.batchNorm, input_channel * 32, input_channel * 16)
        self.pred_mask3 = predict_mask(input_channel * 16, self.assign_ch)

        self.deconv2 = deconv(input_channel * 16, input_channel * 8)
        self.conv2_1 = conv(self.batchNorm, input_channel * 16, input_channel * 8)
        self.pred_mask2 = predict_mask(64, self.assign_ch)

        self.deconv1 = deconv(input_channel * 8, input_channel * 4)
        self.conv1_1 = conv(self.batchNorm, input_channel * 8, input_channel * 4)
        self.pred_mask1 = predict_mask(input_channel * 4, self.assign_ch)

        self.deconv0 = deconv(input_channel * 4, input_channel * 2)
        self.conv0_1 = conv(self.batchNorm, input_channel * 4, input_channel * 2)
        self.pred_mask0 = predict_mask(input_channel * 2, self.assign_ch)

        self.convxa = nn.Conv2d(self.assign_ch, input_channel, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
    def forward(self, x):
        out1 = self.conv0b(self.conv0a(x))  # 5*5
        out2 = self.conv1b(self.conv1a(out1))  # 11*11

        out3 = self.conv2b(self.conv2a(out2))  # 23*23
        out4 = self.conv3b(self.conv3a(out3))  # 47*47
        out5 = self.conv4b(self.conv4a(out4))  # 95*95

        out_deconv3 = self.deconv3(out5)
        b1, c1, h1, w1 = out_deconv3.shape
        b2, c2, h2, w2 = out4.shape
        if h1 != h2:
            out_deconv3 = self.convxa(out_deconv3)
        concat3 = torch.cat((out4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)  # out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        out_conv0_1 = self.conv0_1(concat0)
        mask0 = self.pred_mask0(out_conv0_1)  # TODO try mask0
        # prob0 = self.softmax(mask0)
        prob1 = self.convxa(mask0)

        return prob1
'''
a = torch.ones(3,4)
print (0.5*a)