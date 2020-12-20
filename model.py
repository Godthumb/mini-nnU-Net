import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, n_stage, n_filters_in, n_filters_out, normalization='instancenorm'):
        super(ConvBlock, self).__init__()
        ops = []
        # repeat conv-norm-relu
        for i in range(n_stage):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))  # no change shape after convolution
            # different normalization
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            # for 3D segmentation, batch size maybe set small, so that batchnormalization will be useless
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization == 'none':
                assert False
            ops.append(nn.LeakyReLU(0.01, inplace=True)) # nnU-Net

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, stage, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()
        ops = []
        # conv-norm-relu-conv-norm
        for i in range(stage):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            # for 3D segmentation, batch size maybe set small, so that batchnormalization will be useless
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization == 'none':
                assert False

            if i != stage - 1:
                ops.append(nn.LeakyReLU(0.01, inplace=True))

        self.conv = nn.Sequential(*ops)
        self.leakyrelu = nn.LeakyReLU(0.01, inplace=True)


    def forward(self, x):
        x = self.conv(x) + x
        x = self.leakyrelu(x)
        return x

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()
        ops = []
        # conv-norm
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))  # /2
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
                # for 3D segmentation, batch size maybe set small, so that batchnormalization will be useless
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        # only conv
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.LeakyReLU(0.01, inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()
        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        # change channel but not change shape
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.LeakyReLU(0.01, inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


# when choose stride = 2, better set kernel to even
# here also set kernel = 2
class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


# def upsamping_seg_conv(self, factor_ref_input_seg, n_in_filter):
#     seg_op = []
#     seg_op.append(nn.Conv3d(n_in_filter, self.n_classes, 1, 1, 0, 1, 1, False))  # 1x1x1 conv3d
#     seg_op.append(nn.Upsample(scale_factor=factor_ref_input_seg, mode='trilinear', align_corners=False))
#     conv = nn.Sequential(*seg_op)
#     return nn.ModuleList(conv)

class Upsampling_seg_conv(nn.Module):
    def __init__(self, n_downsampling=4, n_filters=16, n_classes=2):
        super(Upsampling_seg_conv, self).__init__()
        self.ops = []
        for i in range(n_downsampling)[::-1]:  # [3, 2, 1, 0]
            # print(i)
            if i == 0:  # last layer dont't do upsampling
                continue
            seg_op = []
            # 128 -> 2
            seg_op.append(nn.Conv3d(n_filters * (2 ** i), n_classes, 1, 1, 0, 1, 1, False))  # 1x1x1 conv3d
            seg_op.append(nn.Upsample(scale_factor=2**i, mode='trilinear', align_corners=False))
            self.ops.append(nn.Sequential(*seg_op))

        self.ops = nn.ModuleList(self.ops)
    def forward(self, seg_features):
        x = [op(seg_feature) for op, seg_feature in zip(self.ops, seg_features)]
        return x

class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16,
                 normalization='none', has_dropout=False, n_downsampling=4):
        super(VNet, self).__init__()
        self.n_classes = n_classes
        self.has_dropout = has_dropout
        ##############################Downsampling phase######################################
        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        ############################Upsamping phase#########################################################
        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)

        ######################################out phase ##########################################################
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.upsampling_seg_conv = Upsampling_seg_conv(n_downsampling=n_downsampling,
                                                       n_filters=n_filters, n_classes=n_classes)
        self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        deep_supervise_layers = []

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4  # element-wise add

        x6 = self.block_six(x5_up) # deep supervison layer1
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)  # deep supervison layer2
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)  # deep superviosn layer3
        res = [x6, x7, x8, out]
        return res

    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            self.has_dropout = turnoff_drop
        features = self.encoder(input)
        seg_layers = self.decoder(features)
        out = self.upsampling_seg_conv(seg_layers[:-1]) + seg_layers[-1:]
        return out

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == '__main__':
    model = VNet(normalization='instancenorm').cuda()
    inp = torch.randn(1, 1, 128, 128, 128).cuda()
    out = model(inp)
    print([o.shape for o in out])
