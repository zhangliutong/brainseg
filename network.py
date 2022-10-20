from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from packaging import version
import numpy as np
from torch.distributions.normal import Normal

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
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

        ops.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [ torch.arange(0, s) for s in size ] 
        grids = torch.meshgrid(vectors) 
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor)
        #self.register_buffer('grid', grid)

        self.mode = mode
        self.grid = grid

    def forward(self, src, flow):   
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        device = src.device
        self.grid = self.grid.to(device)
        new_locs = self.grid + flow 

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1) 
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1) 
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode)


class Encoder(nn.Module):
    def __init__(self,filters=16,normalization='None'):
        '''
        [16,32,64,64,128]
        '''
        super(Encoder,self).__init__()
        self.block1 = ConvBlock(n_stages=2,n_filters_in=1,n_filters_out=filters,normalization=normalization)
        self.block1_dw = DownsamplingConvBlock(n_filters_in=filters,n_filters_out=filters*2,normalization=normalization)

        self.block2 = ConvBlock(n_stages=2,n_filters_in=filters*2,n_filters_out=filters*2,normalization=normalization)
        self.block2_dw = DownsamplingConvBlock(n_filters_in=filters*2,n_filters_out=filters*4,normalization=normalization)

        self.block3 = ConvBlock(n_stages=2,n_filters_in=filters*4,n_filters_out=filters*4,normalization=normalization)
        self.block3_dw = DownsamplingConvBlock(n_filters_in=filters*4,n_filters_out=filters*4,normalization=normalization)

        self.block4 = ConvBlock(n_stages=2,n_filters_in=filters*4,n_filters_out=filters*4,normalization=normalization)
        self.block4_dw = DownsamplingConvBlock(n_filters_in=filters*4,n_filters_out=filters*8,normalization=normalization)

        self.block5 = ConvBlock(n_stages=1,n_filters_in=filters*8,n_filters_out=filters*8,normalization=normalization)
    
    def forward(self,x):
        x1 = self.block1(x)
        x1_dw = self.block1_dw(x1)

        x2 = self.block2(x1_dw)
        x2_dw = self.block2_dw(x2)

        x3 = self.block3(x2_dw)
        x3_dw = self.block3_dw(x3)

        x4 = self.block4(x3_dw)
        x4_dw = self.block4_dw(x4)

        x5 = self.block5(x4_dw)

        return [x1,x2,x3,x4,x5]

class Decoder(nn.Module):
    def __init__(self,filters=16,normalization='None'):
        super(Decoder,self).__init__()
        self.block4_up =  UpsamplingDeconvBlock(filters * 8, filters * 4, normalization=normalization)
        self.block4 = ConvBlock(n_stages=2,n_filters_in=filters*8,n_filters_out=filters*4,normalization=normalization)
        
        self.block3_up =  UpsamplingDeconvBlock(filters * 4, filters * 4, normalization=normalization)
        self.block3 = ConvBlock(n_stages=2,n_filters_in=filters*8,n_filters_out=filters*4,normalization=normalization)

        self.block2_up =  UpsamplingDeconvBlock(filters * 4, filters * 2, normalization=normalization)
        self.block2 = ConvBlock(n_stages=2,n_filters_in=filters*4,n_filters_out=filters*2,normalization=normalization)

        self.block1_up =  UpsamplingDeconvBlock(filters * 2, filters * 1, normalization=normalization)
        self.block1 = ConvBlock(n_stages=2,n_filters_in=filters*2,n_filters_out=filters*2,normalization=normalization)

        #self.out_conv = nn.Conv3d(filters, 1, 7, padding=3)
        self.out_conv = nn.Conv3d(filters*2, 29, 1)


    def forward(self,features):
        x1,x2,x3,x4,x5 = features
        x4_up = self.block4_up(x5)
        x4_up = torch.cat([x4_up,x4],dim=1)
        x4_up = self.block4(x4_up)

        x3_up = self.block3_up(x4_up)
        x3_up = torch.cat([x3_up,x3],dim=1)
        x3_up = self.block3(x3_up)

        x2_up = self.block2_up(x3_up)
        x2_up = torch.cat([x2_up,x2],dim=1)
        x2_up = self.block2(x2_up)

        x1_up = self.block1_up(x2_up)
        x1_up = torch.cat([x1_up,x1],dim=1)
        x1_up = self.block1(x1_up)

        out = self.out_conv(x1_up)

        return out

def predict_flow(in_planes):
    flow = nn.Conv3d(in_planes,3,kernel_size=3,stride=1,padding=1)
    nd = Normal(0, 1e-5)
    flow.weight = nn.Parameter(nd.sample(flow.weight.shape))
    flow.bias = nn.Parameter(torch.zeros(flow.bias.shape))
    return flow

def up_flow(flow):
    return 2*F.interpolate(flow,scale_factor=2,mode='trilinear')

class Reg(nn.Module):
    def __init__(self,vol_size,normalization='None'):
        '''
        [16,32,64,64,128]
        '''
        super(Reg,self).__init__()
        od = 128+128
        self.block5 = ConvBlock(n_stages=2,n_filters_in=od,n_filters_out=32,normalization=normalization)
        self.block5_up=  UpsamplingDeconvBlock(n_filters_in=32,n_filters_out=8,normalization=normalization)
        self.flo5 = predict_flow(32)

        od = 64+64+8+3
        self.block4 = ConvBlock(n_stages=2,n_filters_in=od,n_filters_out=32,normalization=normalization)
        self.block4_up=  UpsamplingDeconvBlock(n_filters_in=32,n_filters_out=8,normalization=normalization)
        self.flo4 = predict_flow(32)
        self.spa4 = SpatialTransformer([x/8 for x in vol_size])

        od = 64+64+8+3
        self.block3 = ConvBlock(n_stages=2,n_filters_in=od,n_filters_out=32,normalization=normalization)
        self.block3_up=  UpsamplingDeconvBlock(n_filters_in=32,n_filters_out=8,normalization=normalization)
        self.flo3 = predict_flow(32)
        self.spa3 = SpatialTransformer([x/4 for x in vol_size])

        od = 32+32+8+3
        self.block2 = ConvBlock(n_stages=2,n_filters_in=od,n_filters_out=16,normalization=normalization)
        self.block2_up=  UpsamplingDeconvBlock(n_filters_in=16,n_filters_out=8,normalization=normalization)
        self.flo2 = predict_flow(16)
        self.spa2 = SpatialTransformer([x/2 for x in vol_size])

        od =16+16+8+3
        self.block1 = ConvBlock(n_stages=2,n_filters_in=od,n_filters_out=16,normalization=normalization)
        self.flo1 = predict_flow(16)
        self.spa1 = SpatialTransformer([x/1 for x in vol_size])

    
    def forward(self,moving_feats,fixed_feats):
        m1,m2,m3,m4,m5 = moving_feats
        f1,f2,f3,f4,f5 = fixed_feats

        x = torch.cat([m5,f5],dim=1)
        x = self.block5(x)
        flow5 = self.flo5(x)
        up_x5 = self.block5_up(x)
        up_flow5 = up_flow(flow5)

        w4 = self.spa4(m4,up_flow5)
        x = torch.cat([w4,f4,up_x5,up_flow5],dim=1)
        x = self.block4(x)
        flow4 = self.flo4(x)
        up_x4 = self.block4_up(x)
        up_flow4 = up_flow(flow4)

        w3 = self.spa3(m3,up_flow4)
        x = torch.cat([w3,f3,up_x4,up_flow4],dim=1)
        x = self.block3(x)
        flow3 = self.flo3(x)
        up_x3 = self.block3_up(x)
        up_flow3 = up_flow(flow3)

        w2 = self.spa2(m2,up_flow3)
        x = torch.cat([w2,f2,up_x3,up_flow3],dim=1)
        x = self.block2(x)
        flow2 = self.flo2(x)
        up_x2 = self.block2_up(x)
        up_flow2 = up_flow(flow2)

        w1 = self.spa1(m1,up_flow2)
        x = torch.cat([w1,f1,up_x2,up_flow2],dim=1)
        x = self.block1(x)
        flow1 = self.flo1(x)

        return flow1



class Unet(nn.Module):
    def __init__(self,normalization='batchnorm'):
        super(Unet,self).__init__()
        self.encoder = Encoder(normalization=normalization)
        self.decoder = Decoder(normalization=normalization)
    
    def forward(self,x):
        feats = self.encoder(x)
        out = self.decoder(feats)
        return out

class Regnet(nn.Module):
    def __init__(self,shape,normalization='batchnorm'):
        super(Regnet,self).__init__()
        self.encoder = Encoder(normalization=normalization)
        self.reg = Reg(vol_size=shape,normalization=normalization)

    def forward(self,moving,fixed):
        f_feats = self.encoder(fixed)
        m_feats = self.encoder(moving)
        flow = self.reg(m_feats,f_feats)
        warp = self.reg.spa1(moving,flow)
        return warp,flow

class Augnet(nn.Module):
    def __init__(self,shape,normalization='batchnorm'):
        super(Augnet,self).__init__()
        self.encoder = Encoder(normalization=normalization)
        self.reg = Reg(vol_size=shape,normalization=normalization)
        self.decoder = Decoder(normalization=normalization)

    def forward(self):
        pass

