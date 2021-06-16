import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def space_to_depth(x):
    h,w = x.shape[2:4]
    output = torch.cat((x[:,:,0:h:2,0:w:2],
                        x[:,:,0:h:2,1:w:2],
                        x[:,:,1:h:2,1:w:2],
                        x[:,:,1:h:2,0:w:2]),dim=1)
    return output
def upsample(x):
    b,c,h,w = x.shape[0:4]
    avg = nn.AvgPool2d([4,4],stride=4)
    output = avg(x).view(b,c,-1)
    return output

class nonlocalblock(nn.Module):
    def __init__(self,channel=32,avg_kernel=2):
        super(nonlocalblock,self).__init__()
        self.channel = channel//4
        self.theta = nn.Conv2d(channel,self.channel,1)
        self.phi = nn.Conv2d(channel,self.channel,1)
        self.g = nn.Conv2d(channel,self.channel,1)
        self.conv = nn.Conv2d(self.channel,channel,1)
        self.avg = nn.AvgPool2d([avg_kernel,avg_kernel],stride=avg_kernel)
    def forward(self,x):
        H,W = x.shape[2:4]
        #u = F.interpolate(x,scale_factor=0.5)
        #avg = nn.AvgPool2d([2,2],stride=2)
        u=self.avg(x)
        b,c,h,w = u.shape[0:4]
        #avg = nn.AvgPool2d(5,stride=1,padding=2)
        #temp_x = torch.cat((x,avg(x)),dim=1)
        #avg = nn.AvgPool2d(11,stride=1,padding=5)
        #temp_x = torch.cat((temp_x,avg(x)),dim=1)
        theta_x = self.theta(u).view(b,self.channel,-1).permute(0,2,1)
        phi_x = self.phi(u)
        phi_x = upsample(phi_x)
        g_x = self.g(u)
        g_x = upsample(g_x).permute(0,2,1)

        #.view(batch_size,self.channel,-1)
        #theta_x = theta_x.permute(0,2,1)
        theta_x = torch.matmul(theta_x,phi_x)
        theta_x = F.softmax(theta_x,dim=-1)

        y = torch.matmul(theta_x,g_x)
        y = y.permute(0,2,1).contiguous()
        y = y.view(b,self.channel,h,w)
        y = self.conv(y)
        y = F.interpolate(y,size=[H,W])
        return y
        
class seblock(nn.Module):
    def __init__(self,in_channel=32,out_channel=32):
        super(seblock,self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_channel,out_channel),nn.ReLU(),nn.Linear(out_channel,out_channel),nn.Sigmoid())
    def forward(self,x):
        c,h,w = x.shape[1:4]
        avg = nn.AvgPool2d([h,w],stride=1)
        y = avg(x)
        y = y.view(1,1,1,-1)
        y = self.fc(y)
        y = y.view(1,-1,1,1)
        output = x*y
        return output
class Globalblock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Globalblock,self).__init__()
        self.fc = nn.Linear(in_channel,out_channel)
    def forward(self,x):
        c,w,h = x.shape[1:4]
        avg = nn.AvgPool2d([w,h],stride=0)
        x = avg(x)
        x = x.view(1,1,1,-1)
        x = self.fc(x)
        return x
class fusionblock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(fusionblock,self).__init__()
        self.fc = nn.Linear(in_channel,out_channel)
        self.fusion = nn.Sequential(nn.Conv2d(out_channel*2,out_channel,1),nn.LeakyReLU(0.2))
    def forward(self,x,y):
        w,h = y.shape[2:4]
        x = self.fc(x)
        x = x.view(1,-1,1,1)
        x = x.repeat(1,1,w,h)
        x = torch.cat((x,y),dim=1)
        x = self.fusion(x)
        return x
class single_block1(nn.Module): # Mix Attention Block
    def __init__(self,in_channel=32,out_channel=32):
        super(single_block1,self).__init__()
        self.nonlocalblock = nonlocalblock(in_channel)
        self.seblock = seblock(2*in_channel,2*in_channel)
        self.fusion = nn.Sequential(nn.Conv2d(2*in_channel,out_channel,3,padding=1),nn.LeakyReLU(0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1),nn.LeakyReLU(0.2))
        #self.conv2 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1,dilation=1),nn.LeakyReLU(0.2))
    def forward(self,x):
        nonlocal_x = self.nonlocalblock(x)
        #nonlocal_x = self.nonlocalblock(seblock_x)
        x_cat = torch.cat((nonlocal_x,x),dim=1)
        #x_cat = x
        seblock_x = self.seblock(x_cat)
        #x_cat = torch.cat((seblock_x,x),dim=1)
        fusion_x = self.fusion(seblock_x)
        #fusion_x = self.fusion(x_cat)
        conv1 = self.conv1(fusion_x)
        #output = self.conv2(conv1)
        #return output
        return conv1
class single_block(nn.Module):# Channel Attention Block + Conv2d
    def __init__(self,in_channel=32,out_channel=32):
        super(single_block,self).__init__()
        self.conv1 = nn.Sequential(seblock(in_channel,in_channel),nn.Conv2d(in_channel,out_channel,3,padding=1),nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel,out_channel,3,padding=1),nn.LeakyReLU(0.2))
    def forward(self,x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return conv2

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
class ResBlock_CBAM(nn.Module):
    def __init__(self,in_channel, out_channel, stride=1,downsampling=False, expansion = 4):
        super(ResBlock_CBAM,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel*self.expansion),
        )
        self.cbam = CBAM(channel=out_channel*self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        # print("residual:",np.shape(residual))
        out = self.bottleneck(x)
        out = self.cbam(out)
        # print("out:",np.shape(out))
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
class DenseBlock(nn.Module):
    def __init__(self, in_channel=16, out_channel=16):
        super(DenseBlock, self).__init__()
        self.dense_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

        self.dense_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

        self.dense_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        residual = x
        dense_conv1 = self.dense_conv1(x)
        dense_conv2 = self.dense_conv2(dense_conv1)
        dense_conv3 = self.dense_conv3(dense_conv2)
        input_conc = torch.cat((residual,dense_conv3,dense_conv2,dense_conv1),dim=1)
        return input_conc

'''Sony_0509-SE'''# OK
class EnhanceNet(nn.Module):# checkpoint_Sony_0509 / test_results_Sony_0509
    def __init__(self,channel=64):
        super(EnhanceNet,self).__init__()
        ''' Modify the network '''
        # self.att_map_head = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.2))
        # self.att_map_body1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), seblock(64, 64), nn.LeakyReLU(0.2))
        # # self.att_map_body2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, padding=1), nn.LeakyReLU(0.2))
        # # self.att_map_body3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), seblock(64, 64), nn.LeakyReLU(0.2))
        # self.att_map_tail = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1))
        #

        ''' Original network branch '''
        self.inc = single_block(16,32)#512   CAB+Conv
        '''
        self.layer1 = nn.Sequential(nn.Conv2d(32*4,32,1),single_block1(32,64))#256
        self.layer2 = nn.Sequential(nn.Conv2d(64*4,64,1),single_block1(64,128))#128
        self.layer3 = nn.Sequential(nn.Conv2d(128*4,128,1),single_block1(128,256))#64
        '''
        self.layer1 = nn.Sequential(nn.MaxPool2d(2),single_block(32,64))#256  MAB: Mix Attention Block
        self.layer2 = nn.Sequential(nn.MaxPool2d(2),single_block(64,128))#128
        self.layer3 = nn.Sequential(nn.MaxPool2d(2),single_block(128,256))#64
        self.layer4 = nn.Sequential(nn.MaxPool2d(2), single_block(256, 512))  # 32
        #self.inter = nn.Sequential(nn.Conv2d(256*4,256,1),single_block(256,512))
        #self.up0 = nn.ConvTranspose2d(512,256,2,2)
        #self.inter_layer = nn.Sequential(single_block(512,256))

        #self.global_feature = Globalblock(256,256)
        #self.fusionblock0 = fusionblock(256,32)
        #self.fusionblock1 = fusionblock(256,64)
        #self.fusionblock2 = fusionblock(256,128)
        #self.fusionblock3 = fusionblock(256,256)

        self.up1 = nn.ConvTranspose2d(512,256,2,2) # Convtranspose
        self.layer5 = nn.Sequential(single_block(512,256)) # CAB + Conv
        self.up2 = nn.ConvTranspose2d(256,128,2,2)
        self.layer6 = nn.Sequential(single_block(256,128))
        self.up3 = nn.ConvTranspose2d(128,64,2,2)
        self.layer7 = nn.Sequential(single_block(128,64))
        self.up4 = nn.ConvTranspose2d(64,32,2,2)
        self.layer8 = nn.Sequential(single_block(64, 32))
        self.output = nn.Sequential(nn.Conv2d(32,12,1),nn.ReLU())
    def forward(self,x):
        # x = space_to_depth(x)
        ''''''
        '''Modify network backward propogation'''
        # I = torch.cat((0.8 * x, x, 1.2 * x, 1.5 * x), dim=1)
        # att_map_head = self.att_map_head(I) # 32
        # att_map_body1 = self.att_map_body1(att_map_head)#64
        # # att_map_body1 = torch.cat((att_map_head,att_map_body1), dim=1)
        # # att_map_body2 = self.att_map_body2(att_map_body1)
        # # att_map_body2 = torch.cat((att_map_body1,att_map_body2), dim=1)
        # # att_map_body3 = self.att_map_body3(att_map_body2)
        #
        # att_map_tail = self.att_map_tail(att_map_body1)


        ''' Ori network backward propogation'''
        I = torch.cat((0.8*x,x,1.2*x,1.5*x),dim=1)
        inc = self.inc(I)
        # inc = self.inc(att_map_tail)
        '''
        layer1 = self.layer1(space_to_depth(inc))
        layer2 = self.layer2(space_to_depth(layer1))
        layer3 = self.layer3(space_to_depth(layer2))
        '''
        layer1 = self.layer1(inc)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        #global_feature = self.global_feature(layer3)
        #inc = self.fusionblock0(global_feature,inc)
        #layer1 = self.fusionblock1(global_feature,layer1)
        #layer2 = self.fusionblock2(global_feature,layer2)
        #layer3 = self.fusionblock3(global_feature,layer3)
        #inter = self.inter(space_to_depth(layer3))
        #up0 = self.up0(inter)
        #inter_layer = torch.cat((up0,layer3),dim=1)
        #inter_layer = self.inter_layer(inter_layer)
        up1 = self.up1(layer4)
        layer5 = torch.cat((up1,layer3),dim=1)
        layer5 = self.layer5(layer5)
        up2 = self.up2(layer5)
        layer6 = torch.cat((up2,layer2),dim=1)
        layer6 = self.layer6(layer6)
        up3 = self.up3(layer6)
        layer7 = torch.cat((up3,layer1),dim=1)
        layer7 = self.layer7(layer7)
        up4 = self.up4(layer7)
        layer8 = torch.cat((up4,inc),dim=1)
        layer8 = self.layer8(layer8)

        output = self.output(layer8)
        output = F.pixel_shuffle(output,2)  # Conv + Pixel_Shuffle
        return output
'''Sony_0513'''# No
# class EnhanceNet(nn.Module):# checkpoint_Sony_0513 / test_results_Sony_0513
#     def __init__(self,channel=64):
#         super(EnhanceNet,self).__init__()
#         ''' Modify the network '''
#         ''' Original network branch '''
#         self.inc = single_block(16,32)#512   CAB+Conv
#
#         self.layer1 = nn.Sequential(nn.MaxPool2d(2),single_block(32,64),ResBlock_CBAM(64,16))#256  MAB: Mix Attention Block
#         self.layer2 = nn.Sequential(nn.MaxPool2d(2),single_block(64,128),ResBlock_CBAM(128,32))#128
#         self.layer3 = nn.Sequential(nn.MaxPool2d(2),single_block(128,256))#64
#         self.layer4 = nn.Sequential(nn.MaxPool2d(2), single_block(256, 512))  # 32
#         #self.inter = nn.Sequential(nn.Conv2d(256*4,256,1),single_block(256,512))
#         #self.up0 = nn.ConvTranspose2d(512,256,2,2)
#         #self.inter_layer = nn.Sequential(single_block(512,256))
#
#         #self.global_feature = Globalblock(256,256)
#         #self.fusionblock0 = fusionblock(256,32)
#         #self.fusionblock1 = fusionblock(256,64)
#         #self.fusionblock2 = fusionblock(256,128)
#         #self.fusionblock3 = fusionblock(256,256)
#
#         self.up1 = nn.ConvTranspose2d(512,256,2,2) # Convtranspose
#         self.layer5 = nn.Sequential(single_block(512,256)) # CAB + Conv
#         self.up2 = nn.ConvTranspose2d(256,128,2,2)
#         self.layer6 = nn.Sequential(single_block(256,128))
#         self.up3 = nn.ConvTranspose2d(128,64,2,2)
#         self.layer7 = nn.Sequential(single_block(128,64))
#         self.up4 = nn.ConvTranspose2d(64,32,2,2)
#         self.layer8 = nn.Sequential(single_block(64, 32))
#         self.output = nn.Sequential(nn.Conv2d(32,12,1),nn.ReLU())
#     def forward(self,x):
#         # x = space_to_depth(x)
#         ''''''
#         '''Modify network backward propogation'''
#         # I = torch.cat((0.8 * x, x, 1.2 * x, 1.5 * x), dim=1)
#         # att_map_head = self.att_map_head(I) # 32
#         # att_map_body1 = self.att_map_body1(att_map_head)#64
#         # # att_map_body1 = torch.cat((att_map_head,att_map_body1), dim=1)
#         # # att_map_body2 = self.att_map_body2(att_map_body1)
#         # # att_map_body2 = torch.cat((att_map_body1,att_map_body2), dim=1)
#         # # att_map_body3 = self.att_map_body3(att_map_body2)
#         #
#         # att_map_tail = self.att_map_tail(att_map_body1)
#
#
#         ''' Ori network backward propogation'''
#         I = torch.cat((0.8*x,x,1.2*x,1.5*x),dim=1)
#         inc = self.inc(I)
#         # inc = self.inc(att_map_tail)
#         '''
#         layer1 = self.layer1(space_to_depth(inc))
#         layer2 = self.layer2(space_to_depth(layer1))
#         layer3 = self.layer3(space_to_depth(layer2))
#         '''
#         layer1 = self.layer1(inc)
#         layer2 = self.layer2(layer1)
#         layer3 = self.layer3(layer2)
#         layer4 = self.layer4(layer3)
#         #global_feature = self.global_feature(layer3)
#         #inc = self.fusionblock0(global_feature,inc)
#         #layer1 = self.fusionblock1(global_feature,layer1)
#         #layer2 = self.fusionblock2(global_feature,layer2)
#         #layer3 = self.fusionblock3(global_feature,layer3)
#         #inter = self.inter(space_to_depth(layer3))
#         #up0 = self.up0(inter)
#         #inter_layer = torch.cat((up0,layer3),dim=1)
#         #inter_layer = self.inter_layer(inter_layer)
#         up1 = self.up1(layer4)
#         layer5 = torch.cat((up1,layer3),dim=1)
#         layer5 = self.layer5(layer5)
#         up2 = self.up2(layer5)
#         layer6 = torch.cat((up2,layer2),dim=1)
#         layer6 = self.layer6(layer6)
#         up3 = self.up3(layer6)
#         layer7 = torch.cat((up3,layer1),dim=1)
#         layer7 = self.layer7(layer7)
#         up4 = self.up4(layer7)
#         layer8 = torch.cat((up4,inc),dim=1)
#         layer8 = self.layer8(layer8)
#
#         output = self.output(layer8)
#         output = F.pixel_shuffle(output,2)  # Conv + Pixel_Shuffle
#         return output
'''Sony_0516-CA-SPA'''
# class EnhanceNet(nn.Module):# checkpoint_Sony_0516 / test_results_Sony_0516
#     def __init__(self,channel=64):
#         super(EnhanceNet,self).__init__()
#         ''' Modify the network '''
#         ''' Original network branch '''
#         self.inc = single_block(16,32)#512   CAB+Conv
#
#         self.layer1 = nn.Sequential(nn.MaxPool2d(2),single_block(32,64),CBAM(64))#256  MAB: Mix Attention Block
#         self.layer2 = nn.Sequential(nn.MaxPool2d(2),single_block(64,128),CBAM(128))#128
#         self.layer3 = nn.Sequential(nn.MaxPool2d(2),single_block(128,256),CBAM(256))#64
#         self.layer4 = nn.Sequential(nn.MaxPool2d(2), single_block(256, 512),CBAM(512))  # 32
#         #self.inter = nn.Sequential(nn.Conv2d(256*4,256,1),single_block(256,512))
#         #self.up0 = nn.ConvTranspose2d(512,256,2,2)
#         #self.inter_layer = nn.Sequential(single_block(512,256))
#
#         #self.global_feature = Globalblock(256,256)
#         #self.fusionblock0 = fusionblock(256,32)
#         #self.fusionblock1 = fusionblock(256,64)
#         #self.fusionblock2 = fusionblock(256,128)
#         #self.fusionblock3 = fusionblock(256,256)
#
#         self.up1 = nn.ConvTranspose2d(512,256,2,2) # Convtranspose
#         self.layer5 = nn.Sequential(single_block(512,256)) # CAB + Conv
#         self.up2 = nn.ConvTranspose2d(256,128,2,2)
#         self.layer6 = nn.Sequential(single_block(256,128))
#         self.up3 = nn.ConvTranspose2d(128,64,2,2)
#         self.layer7 = nn.Sequential(single_block(128,64))
#         self.up4 = nn.ConvTranspose2d(64,32,2,2)
#         self.layer8 = nn.Sequential(single_block(64, 32))
#         self.output = nn.Sequential(nn.Conv2d(32,12,1),nn.ReLU())
#     def forward(self,x):
#         # x = space_to_depth(x)
#         ''''''
#         '''Modify network backward propogation'''
#         # I = torch.cat((0.8 * x, x, 1.2 * x, 1.5 * x), dim=1)
#         # att_map_head = self.att_map_head(I) # 32
#         # att_map_body1 = self.att_map_body1(att_map_head)#64
#         # # att_map_body1 = torch.cat((att_map_head,att_map_body1), dim=1)
#         # # att_map_body2 = self.att_map_body2(att_map_body1)
#         # # att_map_body2 = torch.cat((att_map_body1,att_map_body2), dim=1)
#         # # att_map_body3 = self.att_map_body3(att_map_body2)
#         #
#         # att_map_tail = self.att_map_tail(att_map_body1)
#
#
#         ''' Ori network backward propogation'''
#         I = torch.cat((0.8*x,x,1.2*x,1.5*x),dim=1)
#         inc = self.inc(I)
#         # inc = self.inc(att_map_tail)
#         '''
#         layer1 = self.layer1(space_to_depth(inc))
#         layer2 = self.layer2(space_to_depth(layer1))
#         layer3 = self.layer3(space_to_depth(layer2))
#         '''
#         layer1 = self.layer1(inc)
#         layer2 = self.layer2(layer1)
#         layer3 = self.layer3(layer2)
#         layer4 = self.layer4(layer3)
#         #global_feature = self.global_feature(layer3)
#         #inc = self.fusionblock0(global_feature,inc)
#         #layer1 = self.fusionblock1(global_feature,layer1)
#         #layer2 = self.fusionblock2(global_feature,layer2)
#         #layer3 = self.fusionblock3(global_feature,layer3)
#         #inter = self.inter(space_to_depth(layer3))
#         #up0 = self.up0(inter)
#         #inter_layer = torch.cat((up0,layer3),dim=1)
#         #inter_layer = self.inter_layer(inter_layer)
#         up1 = self.up1(layer4)
#         layer5 = torch.cat((up1,layer3),dim=1)
#         layer5 = self.layer5(layer5)
#         up2 = self.up2(layer5)
#         layer6 = torch.cat((up2,layer2),dim=1)
#         layer6 = self.layer6(layer6)
#         up3 = self.up3(layer6)
#         layer7 = torch.cat((up3,layer1),dim=1)
#         layer7 = self.layer7(layer7)
#         up4 = self.up4(layer7)
#         layer8 = torch.cat((up4,inc),dim=1)
#         layer8 = self.layer8(layer8)
#
#         output = self.output(layer8)
#         output = F.pixel_shuffle(output,2)  # Conv + Pixel_Shuffle
#         return output
'''Sony_0520_12-DenseBlock'''
# class EnhanceNet(nn.Module):# checkpoint_Sony_0520-No.12 / test_results_Sony_0520-No.12
#     def __init__(self,channel=64):
#         super(EnhanceNet,self).__init__()
#
#
#         ''' Original network branch '''
#         self.input = DenseBlock(16,16)
#
#         self.inc = single_block(64,32)#512   CAB+Conv
#         self.layer1 = nn.Sequential(nn.MaxPool2d(2),single_block(32,64))#256  MAB: Mix Attention Block
#         self.layer2 = nn.Sequential(nn.MaxPool2d(2),single_block(64,128))#128
#         self.layer3 = nn.Sequential(nn.MaxPool2d(2),single_block(128,256))#64
#         self.layer4 = nn.Sequential(nn.MaxPool2d(2), single_block(256, 512))  # 32
#
#         self.up1 = nn.ConvTranspose2d(512,256,2,2) # Convtranspose
#         self.layer5 = nn.Sequential(single_block(512,256)) # CAB + Conv
#         self.up2 = nn.ConvTranspose2d(256,128,2,2)
#         self.layer6 = nn.Sequential(single_block(256,128))
#         self.up3 = nn.ConvTranspose2d(128,64,2,2)
#         self.layer7 = nn.Sequential(single_block(128,64))
#         self.up4 = nn.ConvTranspose2d(64,32,2,2)
#         self.layer8 = nn.Sequential(single_block(64, 32))
#
#
#         self.output = nn.Sequential(nn.Conv2d(32,12,1),nn.ReLU())
#     def forward(self,x):
#
#         I = torch.cat((0.8*x,x,1.2*x,1.5*x),dim=1)
#         # print('shape_I',np.shape(I))
#         input = self.input(I)
#         # print('shape_input',np.shape(input))
#         inc = self.inc(input)
#         # print('shape_inc',np.shape(inc))
#
#
#         layer1 = self.layer1(inc)
#         layer2 = self.layer2(layer1)
#         layer3 = self.layer3(layer2)
#         layer4 = self.layer4(layer3)
#
#         up1 = self.up1(layer4)
#         layer5 = torch.cat((up1,layer3),dim=1)
#         layer5 = self.layer5(layer5)
#         up2 = self.up2(layer5)
#         layer6 = torch.cat((up2,layer2),dim=1)
#         layer6 = self.layer6(layer6)
#         up3 = self.up3(layer6)
#         layer7 = torch.cat((up3,layer1),dim=1)
#         layer7 = self.layer7(layer7)
#         up4 = self.up4(layer7)
#         layer8 = torch.cat((up4,inc),dim=1)
#         # print('shape_layer8-1', np.shape(layer8))
#         layer8 = self.layer8(layer8)
#         # print('shape_layer8-2', np.shape(layer8))
#         output = self.output(layer8)
#         output = F.pixel_shuffle(output,2)  # Conv + Pixel_Shuffle
#         return output

'''Sony_0509-SE_消融实验1（仅使用L1），其他同Sony_0509-SE  '''
# class EnhanceNet(nn.Module):# checkpoint_Sony_0524-L1 / test_results_Sony_0524—L1
#     def __init__(self,channel=64):
#         super(EnhanceNet,self).__init__()
#         ''' Modify the network '''
#         # self.att_map_head = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.2))
#         # self.att_map_body1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), seblock(64, 64), nn.LeakyReLU(0.2))
#         # # self.att_map_body2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, padding=1), nn.LeakyReLU(0.2))
#         # # self.att_map_body3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), seblock(64, 64), nn.LeakyReLU(0.2))
#         # self.att_map_tail = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, padding=1))
#         #
#
#         ''' Original network branch '''
#         self.inc = single_block(16,32)#512   CAB+Conv
#         '''
#         self.layer1 = nn.Sequential(nn.Conv2d(32*4,32,1),single_block1(32,64))#256
#         self.layer2 = nn.Sequential(nn.Conv2d(64*4,64,1),single_block1(64,128))#128
#         self.layer3 = nn.Sequential(nn.Conv2d(128*4,128,1),single_block1(128,256))#64
#         '''
#         self.layer1 = nn.Sequential(nn.MaxPool2d(2),single_block(32,64))#256  MAB: Mix Attention Block
#         self.layer2 = nn.Sequential(nn.MaxPool2d(2),single_block(64,128))#128
#         self.layer3 = nn.Sequential(nn.MaxPool2d(2),single_block(128,256))#64
#         self.layer4 = nn.Sequential(nn.MaxPool2d(2), single_block(256, 512))  # 32
#         #self.inter = nn.Sequential(nn.Conv2d(256*4,256,1),single_block(256,512))
#         #self.up0 = nn.ConvTranspose2d(512,256,2,2)
#         #self.inter_layer = nn.Sequential(single_block(512,256))
#
#         #self.global_feature = Globalblock(256,256)
#         #self.fusionblock0 = fusionblock(256,32)
#         #self.fusionblock1 = fusionblock(256,64)
#         #self.fusionblock2 = fusionblock(256,128)
#         #self.fusionblock3 = fusionblock(256,256)
#
#         self.up1 = nn.ConvTranspose2d(512,256,2,2) # Convtranspose
#         self.layer5 = nn.Sequential(single_block(512,256)) # CAB + Conv
#         self.up2 = nn.ConvTranspose2d(256,128,2,2)
#         self.layer6 = nn.Sequential(single_block(256,128))
#         self.up3 = nn.ConvTranspose2d(128,64,2,2)
#         self.layer7 = nn.Sequential(single_block(128,64))
#         self.up4 = nn.ConvTranspose2d(64,32,2,2)
#         self.layer8 = nn.Sequential(single_block(64, 32))
#         self.output = nn.Sequential(nn.Conv2d(32,12,1),nn.ReLU())
#     def forward(self,x):
#         # x = space_to_depth(x)
#         ''''''
#         '''Modify network backward propogation'''
#         # I = torch.cat((0.8 * x, x, 1.2 * x, 1.5 * x), dim=1)
#         # att_map_head = self.att_map_head(I) # 32
#         # att_map_body1 = self.att_map_body1(att_map_head)#64
#         # # att_map_body1 = torch.cat((att_map_head,att_map_body1), dim=1)
#         # # att_map_body2 = self.att_map_body2(att_map_body1)
#         # # att_map_body2 = torch.cat((att_map_body1,att_map_body2), dim=1)
#         # # att_map_body3 = self.att_map_body3(att_map_body2)
#         #
#         # att_map_tail = self.att_map_tail(att_map_body1)
#
#
#         ''' Ori network backward propogation'''
#         I = torch.cat((0.8*x,x,1.2*x,1.5*x),dim=1)
#         inc = self.inc(I)
#         # inc = self.inc(att_map_tail)
#         '''
#         layer1 = self.layer1(space_to_depth(inc))
#         layer2 = self.layer2(space_to_depth(layer1))
#         layer3 = self.layer3(space_to_depth(layer2))
#         '''
#         layer1 = self.layer1(inc)
#         layer2 = self.layer2(layer1)
#         layer3 = self.layer3(layer2)
#         layer4 = self.layer4(layer3)
#         #global_feature = self.global_feature(layer3)
#         #inc = self.fusionblock0(global_feature,inc)
#         #layer1 = self.fusionblock1(global_feature,layer1)
#         #layer2 = self.fusionblock2(global_feature,layer2)
#         #layer3 = self.fusionblock3(global_feature,layer3)
#         #inter = self.inter(space_to_depth(layer3))
#         #up0 = self.up0(inter)
#         #inter_layer = torch.cat((up0,layer3),dim=1)
#         #inter_layer = self.inter_layer(inter_layer)
#         up1 = self.up1(layer4)
#         layer5 = torch.cat((up1,layer3),dim=1)
#         layer5 = self.layer5(layer5)
#         up2 = self.up2(layer5)
#         layer6 = torch.cat((up2,layer2),dim=1)
#         layer6 = self.layer6(layer6)
#         up3 = self.up3(layer6)
#         layer7 = torch.cat((up3,layer1),dim=1)
#         layer7 = self.layer7(layer7)
#         up4 = self.up4(layer7)
#         layer8 = torch.cat((up4,inc),dim=1)
#         layer8 = self.layer8(layer8)
#
#         output = self.output(layer8)
#         output = F.pixel_shuffle(output,2)  # Conv + Pixel_Shuffle
#         return output