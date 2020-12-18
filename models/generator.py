# -*- coding: utf-8 -*-
"""

"""
import torch.nn as nn
from .layers.helpers import get_nonlinear_layer
from .layers.blocks import Conv2d, UpConv2d

def calc_mean_std(feat, eps=1e-8):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class DownBlock(nn.Module):
    '''
    Down sample block for content and style
    '''
    def __init__(self, in_filters, out_filters, kernel_size=3, stride=1, padding=1,
                 norm_type='instance', nonlinear_type='LeakyReLU'):
        super(DownBlock, self).__init__()
        self.conv1 = Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=1,
                            padding=padding, norm_type=norm_type, nonlinear_type=nonlinear_type)
        self.conv2 = Conv2d(out_filters, out_filters, kernel_size=kernel_size, stride=1,
                            padding=1, norm_type=norm_type, nonlinear_type=None)
        self.downsample = Conv2d(in_filters, in_filters, kernel_size=1, stride=stride,
                                 padding=0, norm_type=norm_type, nonlinear_type=None)
        self.side_conv = Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=1,
                            padding=padding, norm_type=norm_type, nonlinear_type=nonlinear_type)
        self.nonlinear = get_nonlinear_layer(nonlinear_type)

    def forward(self, x):
        down = self.downsample(x)
        
        out = self.conv1(down)
        out = self.conv2(out)
        
        out += self.side_conv(down)
        
        return self.nonlinear(out)
    
class UpBlock(nn.Module):
    '''
    Up sample block for combining content and style
    '''
    def __init__(self, in_filters, out_filters, kernel_size=3, stride=1, padding=1,
                 norm_type='instance', nonlinear_type='LeakyReLU'):
        super(UpBlock, self).__init__()
        self.conv0 = Conv2d(in_filters, in_filters, kernel_size=kernel_size, stride=1,
                            padding=padding, norm_type=norm_type, nonlinear_type=nonlinear_type)
        self.conv1 = Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=1,
                            padding=padding, norm_type=norm_type, nonlinear_type=nonlinear_type)
        self.conv2 = Conv2d(out_filters, out_filters, kernel_size=kernel_size, stride=1,
                            padding=1, norm_type=norm_type, nonlinear_type=None)       
        
        self.side_conv = Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=1,
                            padding=padding, norm_type=norm_type, nonlinear_type=nonlinear_type)
        
        self.upsample = UpConv2d(out_filters, out_filters, kernel_size=kernel_size, scale=2,
                                 padding=padding, norm_type=norm_type)
        
        self.nonlinear = get_nonlinear_layer(nonlinear_type)

    def forward(self, content, style):
        content = self.conv0(content)
        identity = adaptive_instance_normalization(content,style)
        
        out = self.conv1(identity)
        out = self.conv2(out)
        
        out += self.side_conv(identity)
        
        out = self.upsample(out)
        return out


class Generator(nn.Module):
    def __init__(self,opt):
        super(Generator, self).__init__()
        
        n_filters = opt.n_filters       
        #encoder
        self.content_1 = DownBlock(3,n_filters,kernel_size=3, stride=2, norm_type=opt.norm)
        self.content_2 = DownBlock(n_filters, n_filters*2,kernel_size=3, stride=2, norm_type=opt.norm)
        self.content_3 = DownBlock(n_filters*2, n_filters*4,kernel_size=3, stride=2, norm_type=opt.norm)
        self.content_4 = DownBlock(n_filters*4, n_filters*8,kernel_size=3, stride=2, norm_type=opt.norm)
        
        self.style_1 = DownBlock(3,n_filters,kernel_size=3, stride=2, norm_type=opt.norm)
        self.style_2 = DownBlock(n_filters, n_filters*2,kernel_size=3, stride=2, norm_type=opt.norm)
        self.style_3 = DownBlock(n_filters*2, n_filters*4,kernel_size=3, stride=2, norm_type=opt.norm)
        self.style_4 = DownBlock(n_filters*4, n_filters*8,kernel_size=3, stride=2, norm_type=opt.norm)
        #decoder
        self.upblock_1 = UpBlock(n_filters*8, n_filters*4, norm_type=opt.norm)
        self.upblock_2 = UpBlock(n_filters*4, n_filters*2, norm_type=opt.norm)
        self.upblock_3 = UpBlock(n_filters*2, n_filters, norm_type=opt.norm)
        self.upblock_4 = UpBlock(n_filters, n_filters, norm_type=opt.norm)
        
        # Output layers
        output = [nn.Conv2d(n_filters, 3, kernel_size=3, stride=1, padding=1)]

        self.img_out = nn.Tanh()
        self.output = nn.Sequential(*output)

    def forward(self, content, style):
        #encode
        
        content1 = self.content_1(content)
        content2 = self.content_2(content1)
        content3 = self.content_3(content2)
        content4 = self.content_4(content3)
        
        style1 = self.style_1(style)
        style2 = self.style_2(style1)
        style3 = self.style_3(style2)
        style4 = self.style_4(style3)        
        
        #decode
        
        out = self.upblock_1(content4,style4)
        out1 = out.clone() + content3
        out2 = self.upblock_2(out1,style3)
        out3 = out2.clone() + content2
        out4 = self.upblock_3(out3,style2)
        out5 = out4.clone() + content1
        out6 = self.upblock_4(out5,style1)
        
        out7 = self.output(out6)

        return self.img_out(out7)
