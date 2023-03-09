# Taken from DFPNet: https://github.com/thunil/Deep-Flow-Prediction
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0., upsample_factor=2, in_place=True):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=in_place))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=in_place))

    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=upsample_factor, mode='bilinear')) # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))
    return block
    


class DFPNet24(nn.Module):
    def __init__(self, channelExponent=8, dropout=0., in_channels=3, out_channels=1):
        super().__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(in_channels, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout , size=2, pad=2)
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout , size=2, pad=2) 
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=3)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=3,pad=0)
     
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2, pad=1)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=3,pad=0, upsample_factor=1)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout, upsample_factor=1) 
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout, upsample_factor=1 )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, out_channels, 4, 2, 1, bias=True))

    def forward(self, x):
        x_in = x
        if len(x.shape) < 4: 
            x_in = x[:, None]

        out1 = self.layer1(x_in) 
        out2 = self.layer2(out1)   
        out2b= self.layer2b(out2)   
        out3 = self.layer3(out2b)  
        out4 = self.layer4(out3)  
        out5 = self.layer5(out4)  
        out6 = self.layer6(out5)  
        
        dout6 = self.dlayer6(out6)

        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)

        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)

        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)

        if len(x.shape) < 4: 
            dout1 = dout1.squeeze(1)
        return dout1


class DFPNet36(nn.Module):
    def __init__(self, channelExponent=8, dropout=0., in_channels=3, out_channels=1):
        super().__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(in_channels, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout , size=2, pad=2)
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout , size=2, pad=2) 
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=2)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=3,pad=0)
     
        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2, pad=1)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0, upsample_factor=1)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout, upsample_factor=1) 
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout, upsample_factor=1 )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout, size=3, )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, out_channels, 4, 2, 1, bias=True))

    def forward(self, x):
        x_in = x
        if len(x.shape) < 4: 
            x_in = x[:, None]

        out1 = self.layer1(x_in) 
        out2 = self.layer2(out1)   
        out2b= self.layer2b(out2)   
        out3 = self.layer3(out2b)  
        out4 = self.layer4(out3)  
        out5 = self.layer5(out4)  
        out6 = self.layer6(out5)  
        
        dout6 = self.dlayer6(out6)

        
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)

        
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)

        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)

        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)

        if len(x.shape) < 4: 
            dout1 = dout1.squeeze(1)
        return dout1


class DFPNet64(nn.Module):
    def __init__(self, channelExponent=8, dropout=0., in_channels=1, out_channels=1):
        super(DFPNet64, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(in_channels, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout , size=2, pad=2) 
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=2)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=2,pad=0)
     
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0, upsample_factor=1)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout, upsample_factor=1) 
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, out_channels, 4, 2, 1, bias=True))

    def forward(self, x):
        x_in = x
        if len(x.shape) < 4: 
            x_in = x[:, None]

        out1 = self.layer1(x_in) 
        out2 = self.layer2(out1)   
        out2b= self.layer2b(out2)   
        out3 = self.layer3(out2b)  
        out4 = self.layer4(out3)  
        out5 = self.layer5(out4)  
        out6 = self.layer6(out5)  
        dout6 = self.dlayer6(out6)

        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)

        if len(x.shape) < 4: 
            dout1 = dout1.squeeze(1)
        return dout1


class DFPNet100(nn.Module):
    def __init__(self, channelExponent=6, dropout=0., in_channels=1, out_channels=1, in_place=True):
        super(DFPNet100, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(in_channels, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place)
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place, size=4)
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place, size=4)
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place,  size=4 )
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place, size=2,pad=1)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout, in_place=in_place, size=2,pad=1)
     
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout, in_place=in_place, upsample_factor=1, size=4)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout, in_place=in_place, upsample_factor=1, size=3)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout, in_place=in_place ) 
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout, in_place=in_place )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout, in_place=in_place, size=3 )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout, in_place=in_place)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=in_place))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, out_channels, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        
        out2 = self.layer2(out1)
        
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)
        
        out4 = self.layer4(out3)
        

        out5 = self.layer5(out4)
        
        out6 = self.layer6(out4)
        

        dout6 = self.dlayer6(out6)
        
        dout6_out5 = torch.cat([dout6, out5], 1)
        

        dout5 = self.dlayer5(dout6_out5)

        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1


class DFPNet128(nn.Module):
    def __init__(self, channelExponent=6, dropout=0., in_channels=1, out_channels=1, in_place=True):
        super(DFPNet128, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(in_channels, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place)
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place, size=4)
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place, size=4)
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place,  size=4 )
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place, size=2,pad=1)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout, in_place=in_place, size=2,pad=1)
     
        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout, in_place=in_place, upsample_factor=1, size=4)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout, in_place=in_place, upsample_factor=1, size=3)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout, in_place=in_place ) 
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout, in_place=in_place )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout, in_place=in_place )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout, in_place=in_place )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=in_place))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, out_channels, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        
        out2 = self.layer2(out1)
        
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)
        
        out4 = self.layer4(out3)
        

        out5 = self.layer5(out4)
        
        out6 = self.layer6(out4)
        

        dout6 = self.dlayer6(out6)
        
        dout6_out5 = torch.cat([dout6, out5], 1)
        

        dout5 = self.dlayer5(dout6_out5)

        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1


class DFPNet224(nn.Module):
    def __init__(self, channelExponent=6, dropout=0., in_channels=1, out_channels=1, in_place=True):
        super(DFPNet224, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(in_channels, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place )
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place )
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place )
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout , size=4, pad=0, in_place=in_place ) 
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=4,pad=0, in_place=in_place )
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=2,pad=0, in_place=in_place )
     
        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0, upsample_factor=2, in_place=in_place )
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=2, pad=0, upsample_factor=3, in_place=in_place )
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout,  size=6, pad=0, upsample_factor=3, in_place=in_place ) 
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout , in_place=in_place )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout , in_place=in_place )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout , in_place=in_place )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=in_place))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, out_channels, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)

        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)

        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)

        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)

        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = torch.cat([dout2b, out2], 1)

        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        
        dout1 = self.dlayer1(dout2_out1)
        return dout1


class DFPNet384(nn.Module):
    def __init__(self, channelExponent=6, dropout=0., in_channels=1, out_channels=1, in_place=True):
        super(DFPNet384, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(in_channels, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place )
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place )
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place )
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout , size=4, pad=0, in_place=in_place ) 
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=4,pad=0, in_place=in_place )
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=3,pad=0, in_place=in_place )
     
        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0, upsample_factor=4, in_place=in_place )
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=3, pad=0, upsample_factor=3, in_place=in_place )
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout,  size=2, pad=1, upsample_factor=2, in_place=in_place ) 
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout , in_place=in_place )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout , in_place=in_place )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout , in_place=in_place )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=in_place))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, out_channels, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        
        dout6 = self.dlayer6(out6)

        dout6_out5 = torch.cat([dout6, out5], 1)

        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)

        dout4 = self.dlayer4(dout5_out4)

        dout4_out3 = torch.cat([dout4, out3], 1)
        
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)

        dout2b = self.dlayer2b(dout3_out2b)

        dout2b_out2 = torch.cat([dout2b, out2], 1)

        dout2 = self.dlayer2(dout2b_out2)

        dout2_out1 = torch.cat([dout2, out1], 1)
        
        dout1 = self.dlayer1(dout2_out1)
        return dout1


class DFPNet512(nn.Module):
    def __init__(self, channelExponent=6, dropout=0., in_channels=1, out_channels=1, in_place=True):
        super(DFPNet512, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(in_channels, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place )
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place )
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout, in_place=in_place )
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout , size=4, pad=0, in_place=in_place ) # note, size 4!
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=4,pad=0, in_place=in_place )
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=3,pad=0, in_place=in_place )
     
        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0, upsample_factor=3, in_place=in_place )
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=5, pad=0, upsample_factor=3, in_place=in_place )
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout,  size=2, pad=1, upsample_factor=2, in_place=in_place ) 
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout , in_place=in_place )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout , in_place=in_place )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout , in_place=in_place )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=in_place))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, out_channels, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        
        dout6 = self.dlayer6(out6)

        dout6_out5 = torch.cat([dout6, out5], 1)

        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)

        dout4 = self.dlayer4(dout5_out4)

        dout4_out3 = torch.cat([dout4, out3], 1)
        
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)

        dout2b = self.dlayer2b(dout3_out2b)

        dout2b_out2 = torch.cat([dout2b, out2], 1)

        dout2 = self.dlayer2(dout2b_out2)

        dout2_out1 = torch.cat([dout2, out1], 1)
        
        dout1 = self.dlayer1(dout2_out1)
        return dout1


class ScalarFlowUNet384(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, squeeze_last=True, res=None, channelExponent=6, dropout=0.):
        super(ScalarFlowUNet384, self).__init__()
        channels = int(2 ** channelExponent + 0.5)
        self.squeeze_last = squeeze_last

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(in_channels, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout)
        self.layer2b= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout)
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout)
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout , size=4, pad=0) # note, size 4!
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=4,pad=0)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=3,pad=0)

        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0, upsample_factor=4)
        self.dlayer5 = blockUNet(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=3, pad=0, upsample_factor=3)
        self.dlayer4 = blockUNet(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout,  size=2, pad=1, upsample_factor=2) 
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2b= blockUNet(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, out_channels, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b= self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        
        dout6 = self.dlayer6(out6)

        dout6_out5 = torch.cat([dout6, out5], 1)

        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)

        dout4 = self.dlayer4(dout5_out4)

        dout4_out3 = torch.cat([dout4, out3], 1)
        
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)

        dout2b = self.dlayer2b(dout3_out2b)

        dout2b_out2 = torch.cat([dout2b, out2], 1)

        dout2 = self.dlayer2(dout2b_out2)

        dout2_out1 = torch.cat([dout2, out1], 1)
        
        dout1 = self.dlayer1(dout2_out1)

        if self.squeeze_last: dout1 = dout1.squeeze(-3)
        return dout1


def dfpnet_adaptive(res, channel_exponent=6, in_channels=1, out_channels=1, in_place_relu=True):
    "Returns a DFPNet for 2D inputs of the desired resolution `res`."
    if res == 24:
        return DFPNet24(channel_exponent, in_channels=in_channels, out_channels=out_channels)
    elif res == 36:
        return DFPNet36(channel_exponent)
    elif res == 64:
        return DFPNet64(channel_exponent, in_channels=in_channels, out_channels=out_channels)
    elif res == 100:
        return DFPNet100(channel_exponent, in_channels=in_channels, out_channels=out_channels, in_place=False) # Used for T12d on DFP128, should be standardized and added to config
    elif res == 128:
        return DFPNet128(channel_exponent, in_channels=in_channels, out_channels=out_channels, in_place=False)
    elif res == 224:
        return DFPNet224(channel_exponent, in_channels=in_channels, out_channels=out_channels, in_place=in_place_relu)
    elif res == 384:
        return DFPNet384(channel_exponent, in_channels=in_channels, out_channels=out_channels, in_place=in_place_relu)
    elif res == 512:
        return DFPNet512(channel_exponent, in_channels=in_channels, out_channels=out_channels, in_place=in_place_relu)
    else:
        raise NotImplementedError("UNet with current resolution inputs not supported")
        


def blockUNetRectangular(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0., upsample_factor=2, in_place=True, asym_stride=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=in_place))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=in_place))

    if not transposed:
        if asym_stride:
            stride = (2, 1)
        else:
            stride = 2
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        if asym_stride:
            upsample_factor = (2, 1)
        else:
            upsample_factor = 2

        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=upsample_factor, mode='bilinear')) # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))
    return block


class DFPNetScalarFlow(nn.Module):
    def __init__(self, channelExponent=8, dropout=0., in_channels=3, out_channels=1):
        super().__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(in_channels, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNetRectangular(channels  , channels*2, 'layer2', transposed=False, bn=True,  relu=False, dropout=dropout, asym_stride=True)
        self.layer2b= blockUNetRectangular(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )
        self.layer3 = blockUNetRectangular(channels*2, channels*4, 'layer3', transposed=False, bn=True,  relu=False, dropout=dropout , size=2, pad=2)
        self.layer4 = blockUNetRectangular(channels*4, channels*8, 'layer4', transposed=False, bn=True,  relu=False, dropout=dropout , size=2, pad=2) 
        self.layer5 = blockUNetRectangular(channels*8, channels*8, 'layer5', transposed=False, bn=True,  relu=False, dropout=dropout , size=2,pad=3)
        self.layer6 = blockUNetRectangular(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout , size=3,pad=0)
     
        self.dlayer6 = blockUNetRectangular(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout , size=3, pad=1)
        self.dlayer5 = blockUNetRectangular(channels*16,channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout , size=7,pad=0, upsample_factor=1)
        self.dlayer4 = blockUNetRectangular(channels*16,channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout, size=6, pad=0, upsample_factor=2) 
        self.dlayer3 = blockUNetRectangular(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout, upsample_factor=1 )
        self.dlayer2b= blockUNetRectangular(channels*4, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNetRectangular(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, out_channels, (10, 4), 2, 1, bias=True))

    def forward(self, x):
        x_in = x
        if len(x.shape) < 4: 
            x_in = x[:, None]

        out1 = self.layer1(x_in) 
        out2 = self.layer2(out1)   
        out2b= self.layer2b(out2)   
        out3 = self.layer3(out2b)  
        out4 = self.layer4(out3)  
        out5 = self.layer5(out4)  
        out6 = self.layer6(out5)  
        
        dout6 = self.dlayer6(out6)
        
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)

        dout5_out4 = torch.cat([dout5[..., :out4.shape[-2], :], out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        
        dout4_out3 = torch.cat([dout4, out3[..., :dout4.shape[-1]]], 1)
        dout3 = self.dlayer3(dout4_out3)
        
        dout3_out2b = torch.cat([dout3[..., :out2b.shape[-2], :out2b.shape[-1]], out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)

        
        dout2b_out2 = torch.cat([dout2b, out2[..., :dout2b.shape[-2], :]], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1[..., :dout2.shape[-2], :]], 1)
        dout1 = self.dlayer1(dout2_out1)

        if len(x.shape) < 4: 
            dout1 = dout1.squeeze(1)
        return dout1


if __name__ == "__main__":
    x = torch.randn(1, 3, 1062, 600).to("cuda:1")
    m = DFPNetScalarFlow(channelExponent=7, in_channels=3).to("cuda:1")
    y = m(x)
    print(sum([p.numel() for p in m.parameters()]))
    print(y.shape)

    