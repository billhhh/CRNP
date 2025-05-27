import torch
import torch.nn as nn
import torch.nn.functional as F
import Res50

from RNP import RNP


class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                 groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4,
                                                                                                                keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1,
              bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)


def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes, affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg, activation_cfg, kernel_size, stride=(1, 1, 1),
                 padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False):
        super(SeparableConv3d, self).__init__()
        self.depthwise = conv3x3x3(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels, bias=bias, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, in_channels)
        self.pointwise = conv3x3x3(in_channels, out_channels, kernel_size=1, bias=bias, weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.norm1(x)
        x = self.nonlin(x)
        x = self.pointwise(x)
        x = self.norm2(x)
        x = self.nonlin(x)
        return x


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, norm_cfg, activation_cfg, weight_std=False):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, dim_out),
            Activation_layer(activation_cfg, inplace=True),
        )
        self.branch2 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=3, stride=1, padding=2, dilation=2, bias=False,
                      weight_std=weight_std),
            Norm_layer(norm_cfg, dim_out),
            Activation_layer(activation_cfg, inplace=True),
        )
        self.branch3 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=3, stride=1, padding=4, dilation=4, bias=False,
                      weight_std=weight_std),
            Norm_layer(norm_cfg, dim_out),
            Activation_layer(activation_cfg, inplace=True),
        )
        self.branch4 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=3, stride=1, padding=8, dilation=8, bias=False,
                      weight_std=weight_std),
            Norm_layer(norm_cfg, dim_out),
            Activation_layer(activation_cfg, inplace=True),
        )
        self.branch5_conv = conv3x3x3(dim_in, dim_out, kernel_size=1, bias=False, weight_std=weight_std)
        self.branch5_norm = Norm_layer(norm_cfg, dim_out)
        self.branch5_nonlin = Activation_layer(activation_cfg, inplace=True)
        self.conv_cat = nn.Sequential(
            conv3x3x3(dim_out * 5, dim_out, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, dim_out),
            Activation_layer(activation_cfg, inplace=True),
        )

    def forward(self, x):
        [b, c, d, w, h] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, [2, 3, 4], True)
        global_f = global_feature
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_norm(global_feature)
        global_feature = self.branch5_nonlin(global_feature)
        global_feature = F.interpolate(global_feature, (d, w, h), None, 'trilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result, global_f


class SepResBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, weight_std=False):
        super(SepResBlock, self).__init__()
        self.sepconv1 = SeparableConv3d(inplanes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1,
                                        bias=False, weight_std=weight_std)
        self.sepconv2 = SeparableConv3d(planes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1,
                                        bias=False, weight_std=weight_std)
        # self.sepconv3 = SeparableConv3d(planes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)

    def forward(self, x):
        residual = x

        out = self.sepconv1(x)
        out = self.sepconv2(out)
        # out = self.sepconv3(out)
        out = out + residual

        return out


class U_Res3D(nn.Module):
    def __init__(self, norm_cfg='BN', activation_cfg='LeakyReLU', num_classes=None, weight_std=False):
        super(U_Res3D, self).__init__()

        self.MODEL_NUM_CLASSES = num_classes

        self.asppreduce = nn.Sequential(
            conv3x3x3(1280, 256, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 256),
            Activation_layer(activation_cfg, inplace=True),
        )

        self.aspp = ASPP(256, 256, norm_cfg, activation_cfg, weight_std=weight_std)

        self.upsamplex2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')

        self.shortcut_conv3 = nn.Sequential(
            conv3x3x3(256 * 16, 256, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 256),
            Activation_layer(activation_cfg, inplace=True),
        )

        self.shortcut_conv2 = nn.Sequential(
            conv3x3x3(128 * 16, 128, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 128),
            Activation_layer(activation_cfg, inplace=True),
        )

        self.shortcut_conv1 = nn.Sequential(
            conv3x3x3(64 * 16, 64, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 64),
            Activation_layer(activation_cfg, inplace=True),
        )

        self.shortcut_conv0 = nn.Sequential(
            conv3x3x3(64 * 4, 32, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 32),
            Activation_layer(activation_cfg, inplace=True),
        )

        self.transposeconv_stage3 = nn.ConvTranspose3d(256, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage2 = nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)

        self.transposeconv_stage3_partial = nn.ConvTranspose3d(256 * 4, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)

        # self.stage2_de = SepResBlock(256, 256, norm_cfg, activation_cfg, weight_std=weight_std)
        # self.stage1_de = SepResBlock(128, 128, norm_cfg, activation_cfg, weight_std=weight_std)
        # self.stage0_de = SepResBlock(32, 32, norm_cfg, activation_cfg, weight_std=weight_std)

        self.stage3_de = Res50.BasicBlock(256, 256, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage2_de = Res50.BasicBlock(128, 128, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage1_de = Res50.BasicBlock(64, 64, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage0_de = Res50.BasicBlock(32, 32, norm_cfg, activation_cfg, weight_std=weight_std)

        self.cls_conv = nn.Sequential(
            nn.Conv3d(32, self.MODEL_NUM_CLASSES, kernel_size=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd, nn.ConvTranspose3d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.backbone = Res50.ResNet(depth=50, shortcut_type='B', norm_cfg=norm_cfg, activation_cfg=activation_cfg,
                                     weight_std=weight_std)

        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        _ = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_reducechannel = self.asppreduce(layers[-1])
        feature_aspp, global_f = self.aspp(feature_reducechannel)  # 256

        self.x_aspp = feature_aspp
        self.x_global_f = global_f

        return 0

    def forward_partial(self, feature_aspp, layers):
        # x = self.transposeconv_stage3(feature_aspp)  # 256
        x = self.transposeconv_stage3_partial(feature_aspp)  # 512
        skip3 = self.shortcut_conv3(layers[-2])
        x = x + skip3
        x = self.stage3_de(x)

        x = self.transposeconv_stage2(x)
        skip2 = self.shortcut_conv2(layers[-3])
        x = x + skip2
        x = self.stage2_de(x)

        x = self.transposeconv_stage1(x)
        skip1 = self.shortcut_conv1(layers[-4])
        x = x + skip1
        x = self.stage1_de(x)

        x = self.transposeconv_stage0(x)
        skip0 = self.shortcut_conv0(layers[-5])
        x = x + skip0
        x = self.stage0_de(x)

        logits = self.cls_conv(x)
        logits = self.upsamplex2(logits)
        return [logits]


class CRNP(nn.Module):
    """
    SingleNet
    """

    def __init__(self, args, norm_cfg='BN', activation_cfg='LeakyReLU', num_classes=None,
                 weight_std=False, self_att=False):
        super().__init__()
        self.do_ds = False
        self.seg_3D_net = U_Res3D(norm_cfg, activation_cfg, num_classes, weight_std)

        d, h, w = map(int, args.input_size.split(','))
        embed_dim = 125
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=5).cuda()
        self.self_att = self_att

        self.flair_r_model = RNP(in_c=256, out_c=256, USE_GPU=False)
        self.t1_r_model = RNP(in_c=256, out_c=256, USE_GPU=False)
        self.t1ce_r_model = RNP(in_c=256, out_c=256, USE_GPU=False)
        self.t2_r_model = RNP(in_c=256, out_c=256, USE_GPU=False)

    def getGapMap(self, modality, rmodel_robin, ft_robin):

        gap_map = 0
        for i, i_rmodel in enumerate(rmodel_robin):
            if i == modality:
                continue

            # sttADD
            gap_map = gap_map + i_rmodel.eval_model(ft_robin[i].detach().cpu().numpy()).cuda()
        return gap_map

    def forward(self, images, val=False):
        # image = np.array([flair, t1, t1ce, t2]) a = torch.tensor([[1,2,3,4],[5,6,7,8]])
        N, C, D, W, H = images.shape

        x = images.view(N*C, 1, D, W, H)
        _ = self.seg_3D_net(x)
        x_ft = self.seg_3D_net.x_aspp
        x_layers = self.seg_3D_net.backbone.get_layers()
        x_layers_trans = [i_layer.view(N, C * i_layer.shape[1], i_layer.shape[2], i_layer.shape[3], i_layer.shape[4]) for i_layer in x_layers]

        flair_ft = x_ft[0:N*C+1:C]
        t1_ft = x_ft[1:N*C+1:C]
        t1ce_ft = x_ft[2:N*C+1:C]
        t2_ft = x_ft[3:N*C+1:C]

        # calculate gap maps
        rmodel_robin = [self.flair_r_model, self.t1_r_model, self.t1ce_r_model, self.t2_r_model]
        ft_robin = [flair_ft, t1_ft, t1ce_ft, t2_ft]
        flair_gap_map = self.getGapMap(0, rmodel_robin, ft_robin)
        t1_gap_map = self.getGapMap(1, rmodel_robin, ft_robin)
        t1ce_gap_map = self.getGapMap(2, rmodel_robin, ft_robin)
        t2_gap_map = self.getGapMap(3, rmodel_robin, ft_robin)

        org_size = flair_gap_map.shape
        flair_gap_map = flair_gap_map.view(org_size[0], org_size[1], -1)
        flair_gap_map = F.normalize(flair_gap_map, dim=2).view(org_size)
        org_size = t1_gap_map.shape
        t1_gap_map = t1_gap_map.view(org_size[0], org_size[1], -1)
        t1_gap_map = F.normalize(t1_gap_map, dim=2).view(org_size)
        org_size = t1ce_gap_map.shape
        t1ce_gap_map = t1ce_gap_map.view(org_size[0], org_size[1], -1)
        t1ce_gap_map = F.normalize(t1ce_gap_map, dim=2).view(org_size)
        org_size = t2_gap_map.shape
        t2_gap_map = t2_gap_map.view(org_size[0], org_size[1], -1)
        t2_gap_map = F.normalize(t2_gap_map, dim=2).view(org_size)

        x_ft2 = x_ft.clone()
        x_ft2[0:N*C+1:C] = flair_ft * flair_gap_map + flair_ft
        x_ft2[1:N*C+1:C] = t1_ft * t1_gap_map + t1_ft
        x_ft2[2:N*C+1:C] = t1ce_ft * t1ce_gap_map + t1ce_ft
        x_ft2[3:N*C+1:C] = t2_ft * t2_gap_map + t2_ft

        if self.self_att:
            # self attention
            cat_attend = x_ft2.view(N, C * x_ft2.shape[1], x_ft2.shape[2], x_ft2.shape[3], x_ft2.shape[4])
            original_size = cat_attend.size()
            flat_input = cat_attend.view(original_size[0], original_size[1],
                                         original_size[2] * original_size[3] * original_size[4])
            perm_input = flat_input.permute(1, 0, 2)

            att_input, att_weights = self.multihead_attn(perm_input, perm_input, perm_input)
            flat_output = att_input.permute(1, 0, 2)
            out_attend = flat_output.view(original_size)

            logits = self.seg_3D_net.forward_partial(out_attend, x_layers_trans)
        else:
            cat_attend = x_ft2.view(N, C * x_ft2.shape[1], x_ft2.shape[2], x_ft2.shape[3], x_ft2.shape[4])
            logits = self.seg_3D_net.forward_partial(cat_attend, x_layers_trans)

        # training rmodels
        if not val:
            for _ in range(3):
                flair_gap_loss = self.flair_r_model.train_model(flair_ft.detach().cpu().numpy())
                t1_gap_loss = self.t1_r_model.train_model(t1_ft.detach().cpu().numpy())
                t1ce_gap_loss = self.t1ce_r_model.train_model(t1ce_ft.detach().cpu().numpy())
                t2_gap_loss = self.t2_r_model.train_model(t2_ft.detach().cpu().numpy())

        return logits[0]
