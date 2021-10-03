import math

# from os.path import join as 
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)



class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, bn_pretrain=False):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()
        self.bn_pretrain = bn_pretrain

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        
        if bn_pretrain:
            self.bn1 = nn.BatchNorm2d(out_filters)
        else:
            self.bn1_ = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        
        if bn_pretrain:
            self.bn2 = nn.BatchNorm2d(out_filters)
        else:
            self.bn2_ = nn.BatchNorm2d(out_filters)


    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        
        if self.bn_pretrain:
            resA1 = self.bn1(resA)
        else:
            resA1 = self.bn1_(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        
        if self.bn_pretrain:
            resA2 = self.bn2(resA)
        else:
            resA2 = self.bn2_(resA)

        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, bn_pretrain=False, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.bn_pretrain = bn_pretrain
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        
        if bn_pretrain:
            self.bn1 = nn.BatchNorm2d(out_filters)
        else:
            self.bn1_ = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        
        if bn_pretrain:
            self.bn2 = nn.BatchNorm2d(out_filters)
        else:
            self.bn2_ = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        
        if bn_pretrain:
            self.bn3 = nn.BatchNorm2d(out_filters)
        else:
            self.bn3_ = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        
        if bn_pretrain:
            self.bn4 = nn.BatchNorm2d(out_filters)
        else:
            self.bn4_ = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        
        if self.bn_pretrain:
            resA1 = self.bn1(resA)
        else:
            resA1 = self.bn1_(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        
        if self.bn_pretrain:
            resA2 = self.bn2(resA)
        else:
            resA2 = self.bn2_(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        
        if self.bn_pretrain:
            resA3 = self.bn3(resA)
        else:
            resA3 = self.bn3_(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        
        if self.bn_pretrain:
            resA = self.bn4(resA)
        else:
            resA = self.bn4_(resA)
        resA = shortcut + resA


        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB

class ResNetV2(nn.Module):
    def __init__(self, bn_pretrain):
        super(ResNetV2, self).__init__()

        self.downCntx = ResContextBlock(5, 32,bn_pretrain)
        self.downCntx2 = ResContextBlock(32, 32,bn_pretrain)
        self.downCntx3 = ResContextBlock(32, 32,bn_pretrain)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2,bn_pretrain=bn_pretrain, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2,bn_pretrain=bn_pretrain, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2,bn_pretrain=bn_pretrain, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2,bn_pretrain=bn_pretrain, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2,bn_pretrain=bn_pretrain, pooling=False)
        
        self.out_width = 2 * 4 * 32


    def forward(self, x):
        features = []
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        #first after pooling -- second before pooling
        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)  
        
        features.append(down3b)
        features.append(down2b)
        features.append(down1b)
        features.append(down0b)
        
        return down5c, features


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1,dropout_rate=0.2, eval_uncer=False):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has the stride it on conv1!!->wout = (win +1)/2
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)
            
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.eval_uncer=eval_uncer

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        if self.eval_uncer:
            y = self.dropout(y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[(n_block+ '/'+ n_unit+ "/conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[(n_block+ '/'+ n_unit+ "/conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[(n_block+ '/'+ n_unit+ "/conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[(n_block+ '/'+ n_unit+ "/gn1/scale")])
        gn1_bias = np2th(weights[(n_block+ '/'+ n_unit+ "/gn1/bias")])

        gn2_weight = np2th(weights[(n_block+ '/'+ n_unit+ "/gn2/scale")])
        gn2_bias = np2th(weights[(n_block+ '/'+ n_unit+ "/gn2/bias")])

        gn3_weight = np2th(weights[(n_block+ '/'+ n_unit+ "/gn3/scale")])
        gn3_bias = np2th(weights[(n_block+ '/'+ n_unit+ "/gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[(n_block+ '/'+  n_unit+ "/conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[(n_block+ '/'+ n_unit+ "/gn_proj/scale")])
            proj_gn_bias = np2th(weights[(n_block+ '/'+ n_unit+ "/gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

        
class ResNetV2_transunet(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor,drp_out_rate=0.2,eval_uncer_f=False):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        #Wout after the root =(win+1)/2
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(5, width, kernel_size=7, stride=2, bias=False, padding=3)), #original input channel 3
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict( #wout=win
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width, dropout_rate=drp_out_rate, eval_uncer=eval_uncer_f))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width, dropout_rate=drp_out_rate, eval_uncer=eval_uncer_f)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict( #wout = (win +1)/2
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2, dropout_rate=drp_out_rate, eval_uncer=eval_uncer_f))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2, dropout_rate=drp_out_rate, eval_uncer=eval_uncer_f)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict( #wout = (win +1)/2
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2, dropout_rate=drp_out_rate,eval_uncer=eval_uncer_f))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4, dropout_rate=drp_out_rate, eval_uncer=eval_uncer_f)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size_row, in_size_col = x.size()
        # print("x before root")
        # print(x.size())
        x = self.root(x) #after root h, w are divided by 2 #Wout after the root =(win+1)/2
        # print("x after root")
        # print(x.size())
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)#after max pool h, w are divided by 2-->wout=(win-1)/2
        # print("x after max pool")
        # print(x.size())
        for i in range(len(self.body)-1): #-1 as range stats from 0 and here he excludes the last one
            x = self.body[i](x)
            right_size_row = int((in_size_row / 4 / (i+1)))# after 1st body h,w divided by 1 , after 2nd body h,w divided by 2 (as if seems that every one time, body size should be divided by 4) 
            right_size_col = int((in_size_col / 4 / (i+1)))
            # print("x.size()[2]")
            # print(x.size()[2])
            # print("right_size")
            # print(right_size)
            if x.size()[2] != right_size_row or x.size()[3] != right_size_col:
                if x.size()[2] != right_size_row :
                    pad_row = right_size_row - x.size()[2]
                    assert pad_row < 3 and pad_row > 0, "x {} should {}".format(x.size(), right_size_row)
                if x.size()[3] != right_size_col:
                    pad_col = right_size_col - x.size()[3]
                    assert pad_col < 3 and pad_col > 0, "x {} should {}".format(x.size(), right_size_col)
                feat = torch.zeros((b, x.size()[1], right_size_row, right_size_col), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            
            # print("feat")
            # print(feat.size())
            features.append(feat)#except the output of the last body becuase it;s the one that gives the latent var to the decoder
        x = self.body[-1](x)
        # print("x after last body")
        # print(x.size())
        return x, features[::-1]#reverses features list
