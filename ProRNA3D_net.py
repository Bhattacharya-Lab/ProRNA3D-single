##################################################################################################################################
# (Partially modified) ResNet block and Triangle-aware attention block is taken from http://huanglab.phys.hust.edu.cn/DeepInter/
# Credit: Protein–protein contact prediction by geometric triangle-aware protein language models by Lin et. al.
#          nature machine intelligence, 2023
##################################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from egnn_clean import *
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

""" conv1d """
def conv1d( in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int = 1, 
            padding: str = "same", 
            dilation: int = 1, 
            group: int = 1, 
            bias: bool = False) -> nn.Conv1d:

    if padding == "same":
        padding = int((kernel_size - 1)/2)

    return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)

""" conv2d """
def conv2d(in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int = 1, 
            padding: str = "same", 
            dilation: int = 1, 
            group: int = 1, 
            bias: bool = False) -> nn.Conv2d:

    if padding == "same":
        padding = int((kernel_size - 1)/2)

    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)

""" conv1d 1x1 """
def conv_identity_1d( in_channels  : int,
                      out_channels : int,
                      kernel_size  : int = 1,
                      stride       : int = 1,
                      padding      : str = "same",
                      dilation     : int = 1,
                      group        : int = 1,
                      bias         : bool = False,
                      norm         : str = "IN",
                      activation   : str = "Relu",
                      track_running_stats_ : bool = True):
    layers = []

    # convolution
    layers.append( conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias))

    # normalization
    if norm == "BN":
       layers.append( nn.BatchNorm1d(out_channels, affine=True, track_running_stats=track_running_stats_))
    elif norm == "IN":
        layers.append( nn.InstanceNorm1d(out_channels, affine=True, track_running_stats=track_running_stats_))
       
    # activation
    if activation == "ELU":
        layers.append( nn.ELU())
    elif activation == "Relu":
        layers.append(nn.LeakyReLU(negative_slope=0.01,inplace=True))

    return nn.Sequential(*layers)

""" conv2d 1x1"""
def conv_identity_2d( in_channels  : int,
                      out_channels : int,
                      kernel_size  : int = 1,
                      stride       : int = 1,
                      padding      : str = "same",
                      dilation     : int = 1,
                      group        : int = 1,
                      bias         : bool = False,
                      norm         : str = "IN",
                      activation   : str = "Relu",
                      track_running_stats_ : bool = True):
    layers = []

    # convolution
    layers.append(conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias))

    # normalization
    if norm == "BN":
       layers.append( nn.BatchNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_))
    elif norm == "IN":
        layers.append( nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_))

    # activation
    if activation == "ELU":
        layers.append( nn.ELU())
    elif activation == "Relu":
        layers.append( nn.LeakyReLU(negative_slope=0.01,inplace=True))

    return nn.Sequential(*layers)




""" ResNetv2 BasicBlock1D """
class BasicBlock_ResNetV2_1D(nn.Module):

    def __init__(self,
        in_channels  : int,
        out_channels : int,
        kernel_size  : int,
        stride       : int = 1,
        downsample = None,
        padding      : str = "same",
        dilation     : int = 1,
        group        : int = 1,
        bias         : bool = False,
        track_running_stats_ : bool = True,
        norm         : str = "BN",
        activation   : str = "ELU"):

        super(BasicBlock_ResNetV2_1D, self).__init__()

        if norm == "BN":
            self.bn1 = nn.BatchNorm1d(in_channels, affine=True, track_running_stats=track_running_stats_)
            self.bn2 = nn.BatchNorm1d(out_channels, affine=True, track_running_stats=track_running_stats_)
        elif norm == "IN":
            self.bn1 = nn.InstanceNorm1d(in_channels, affine=True, track_running_stats=track_running_stats_)
            self.bn2 = nn.InstanceNorm1d(out_channels, affine=True, track_running_stats=track_running_stats_)

        if activation == "ELU":
            self.relu1 = nn.ELU()
            self.relu2 = nn.ELU()
        elif activation == "Relu":
            self.relu1 = nn.LeakyReLU(negative_slope=0.01,inplace=True)
            self.relu2 = nn.LeakyReLU(negative_slope=0.01,inplace=True)

        self.conv1 = conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)
        self.conv2 = conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)

        self.downsample = downsample

    def forward(self, x):

        identity = x

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.downsample != None :
            identity = self.downsample(identity)

        x += identity

        return x


class BasicBlock_Inception2D_V1(nn.Module):

    def __init__(self,
        in_channels  : int,
        out_channels : int,
        kernel_size  : int,
        stride       : int = 1,
        downsample = None,
        padding      : str = "same",
        dilation     : int = 1,
        group        : int =1,
        bias         : bool = False,
        track_running_stats_ : bool = True,
        norm         : str = "IN",
        activation   : str = "Relu"):

        super(BasicBlock_Inception2D_V1, self).__init__()

        if norm == "BN":
            self.bns1 = nn.ModuleList( [ nn.BatchNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_) for _ in range(3) ])
            self.bns2 = nn.ModuleList( [ nn.BatchNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_) for _ in range(3) ])
        elif norm == "IN":
            self.bns1 = nn.ModuleList( [ nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_) for _ in range(3)])
            self.bns2 = nn.ModuleList( [ nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=track_running_stats_) for _ in range(3)])

        if activation == "ELU":
            self.acts1 = nn.ModuleList( [ nn.ELU() for _ in range(3) ] )
            self.act = nn.ELU()
        elif activation == "Relu":
            self.acts1 = nn.ModuleList( [ nn.LeakyReLU(negative_slope=0.01,inplace=True)  for _ in range(3) ] )
            self.act = nn.LeakyReLU(negative_slope=0.01,inplace=True)

        self.convs1 = nn.ModuleList( [ nn.Conv2d(in_channels, out_channels, (1,9), stride, (0,4), dilation, group, bias),\
                                      nn.Conv2d(in_channels, out_channels, (9,1), stride, (4,0), dilation, group, bias),\
                                      nn.Conv2d(in_channels, out_channels, (3,3), stride, (1,1), dilation, group, bias) ] )

        self.convs2 = nn.ModuleList( [ nn.Conv2d(out_channels, out_channels, (1,9), stride, (0,4), dilation, group, bias),\
                                      nn.Conv2d(out_channels, out_channels, (9,1), stride, (4,0), dilation, group, bias),\
                                      nn.Conv2d(out_channels, out_channels, (3,3), stride, (1,1), dilation, group, bias) ] )

        self.downsample = downsample


    def forward(self, x):

        identity = x

        xs = None
        for i in range(3):

            xsi = self.convs1[i](x)
            xsi = self.bns1[i](xsi)
            xsi = self.acts1[i](xsi)

            xsi = self.convs2[i](xsi)
            xsi = self.bns2[i](xsi)

            if xs == None:
                xs = xsi
            else:
                xs = xs + xsi

        if self.downsample != None:
            identity = self.downsample(identity)

        return self.act(xs + identity)


""" concatenate 1D -> 2D """
def seq2pairwise_v3(rec1d, lig1d):

    device = rec1d.device
    b, c, L1 = rec1d.size()
    _, _, L2 = lig1d.size()

    out1 = rec1d.unsqueeze(3).to(device)
    repeat_idx = [1] * out1.dim()
    repeat_idx[3] = L2
    out1 = out1.repeat(*(repeat_idx))

    out2 = lig1d.unsqueeze(2).to(device)
    repeat_idx = [1] * out2.dim()
    repeat_idx[2] = L1
    out2 = out2.repeat(*(repeat_idx))

    return torch.cat([out1, out2], dim=1)


#####################################################################################################################

class Res_middle(nn.Module):

    def __init__(self):
        super(Res_middle, self).__init__()

        #args1d =model_args['BasicBlock1D']
        args2d = {
                "name"        : BasicBlock_Inception2D_V1,
                "InChannels"  : 144+2,
                "Channels"    : [64 for i in range(4)],
                "OutChannels" : 64,
                "num_Cycle"   : 1,
                "Kernel_size" : 3,
                "Dilation"    : [1],
                "Group"       : 1,
                "Bias"        : False,
                "track_running_stats" : False
            }

        #self.identity1 = conv_identity_2d(args1d['InChannels']*2, args1d['Channels'][0], 1, 1, bias=False)
        self.identity2 = conv_identity_2d(args2d['InChannels'], args2d['Channels'][0], 1, 1, bias=False)
        self.identity3 = conv_identity_2d(args2d['Channels'][0]*2, args2d['Channels'][0], 1, 1, bias=False)

        self.layer2 = self._make_layer(conv2d)

        # output
        #if model_args['dist'] == True:
        #    self.conv = conv2d( args2d['Channels'][-1], model_args['dist_bins'], 1, 1)
        #    self.acti = nn.Softmax(1)
        #else:
        self.conv = conv2d( args2d['Channels'][-1], 1, 1, 1)
        self.acti = nn.Sigmoid()
        #self.acti = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)


    # downsample
    def _downsample(self, conv, in_channels, out_channels, stride):

        if in_channels == out_channels and stride == 1 :
            return None
        else :
            return nn.Sequential( conv(in_channels, out_channels, kernel_size=1, stride=stride) )

    # make layers
    def _make_layer(self, fn):

        conv         = fn
        Block        = BasicBlock_Inception2D_V1
        Num_Blocks   = 20#len(config['Channels'])
        Block_Cycle  = 1
        in_channels  = 64
        out_channels = [64 for i in range(Num_Blocks)]
        kernel_size  = 3
        dilations    = [1]
        group        = 1
        bias         = False
        track_running_stats = False
        stride = 1
        padding = "same"

        layers = []
        for i in range(Num_Blocks):

            n_dilation = len(dilations)
            dilation = dilations[ i % n_dilation]

            if i == 0:
                downsample = self._downsample(conv, in_channels, out_channels[0], stride)
                layers.append( Block(in_channels, out_channels[0], kernel_size, stride, downsample, padding, dilation, group, bias, track_running_stats) )

                for j in range(1, Block_Cycle):
                    layers.append( Block(out_channels[0], out_channels[0], kernel_size, stride, None, padding, dilation, group, bias, track_running_stats) )
            else :
                downsample = self._downsample(conv, out_channels[i-1], out_channels[i], stride)
                layers.append( Block(out_channels[i-1], out_channels[i], kernel_size, stride, downsample, padding, dilation, group, bias, track_running_stats) )

                for j in range(1, Block_Cycle):
                    layers.append( Block(out_channels[i], out_channels[i], kernel_size, stride, None, padding, dilation, group, bias, track_running_stats) )

        return nn.Sequential(*layers)

    def forward(self, com2d):

        #pair1 = seq2pairwise_v3(rec1d, lig1d)
        #pair1 = self.identity1(pair1)

        #pair2 = self.identity2(com2d)
        #pair = torch.cat([pair1, pair2], dim=1)
        #pair = self.identity3(pair)

        out = self.layer2(com2d)
        out_act = self.conv(out)
        out_act = self.acti(out_act)

        return out, out_act
#####################################################################################################################

class TriangleMultiplication(nn.Module):
    def __init__(self):
        super(TriangleMultiplication, self).__init__()

        self.dz = 64# model_args['Channel_z']
        self.dc = 64#model_args['Channel_z']

        # init norm
        self.norm_com = nn.LayerNorm(self.dz)
        self.norm_rec = nn.LayerNorm(self.dz)
        self.norm_lig = nn.LayerNorm(self.dz)

        # linear * gate for com_rec, com_lig
        self.Linear_com_rec = nn.Linear(self.dz, self.dc)
        self.Linear_com_lig = nn.Linear(self.dz, self.dc)
        self.gate_com_rec = nn.Linear(self.dz, self.dc)
        self.gate_com_lig = nn.Linear(self.dz, self.dc)

        # linear * gate for rec, lig
        self.Linear_rec = nn.Linear(self.dz, self.dc)
        self.Linear_lig = nn.Linear(self.dz, self.dc)
        self.gate_rec = nn.Linear(self.dz, self.dc)
        self.gate_lig = nn.Linear(self.dz, self.dc)

        # final output
        self.norm_all = nn.LayerNorm(self.dc)
        self.Linear_all = nn.Linear(self.dc, self.dz)
        self.gate_all = nn.Linear(self.dz, self.dz)

    def forward(self, z_com, z_rec, z_lig, mask=None, mask_sa=None):
        """
        Argument:
            z_com : (B, nrec, nlig, dz)
            z_rec : (B, nrec, nrec, dz)
            z_lig : (B, nlig, nlig, dz)
            mask  : (B, nrec, nlig)
        return:
            z_com : (B, nrec, nlig, dz)
        """
        z_com = self.norm_com(z_com)
        z_rec = self.norm_rec(z_rec)
        z_lig = self.norm_lig(z_lig)
        z_com_init = z_com

        if mask != None:
            z_com_rec = self.Linear_com_rec(z_com) * \
                        ( self.gate_com_rec(z_com).sigmoid() * mask)
            z_com_lig = self.Linear_com_lig(z_com) * \
                        ( self.gate_com_lig(z_com).sigmoid() * mask)
        else:
            z_com_rec = self.Linear_com_rec(z_com) * \
                        ( self.gate_com_rec(z_com).sigmoid())
            z_com_lig = self.Linear_com_lig(z_com) * \
                        ( self.gate_com_lig(z_com).sigmoid())

        if mask_sa != None:
            z_com_rec = z_com_rec * mask_sa
            z_com_lig = z_com_lig * mask_sa


        z_rec = self.Linear_rec(z_rec) * self.gate_rec(z_rec).sigmoid()
        z_lig = self.Linear_lig(z_lig) * self.gate_lig(z_lig).sigmoid()

        z_com_rec = torch.einsum(f"bikc,bkjc->bijc", z_rec, z_com_rec)
        z_com_lig = torch.einsum(f"bikc,bjkc->bjic", z_lig, z_com_lig)
        z_all = z_com_rec + z_com_lig

        z_com = self.gate_all(z_com_init).sigmoid() * self.Linear_all( self.norm_all(z_all))

        return z_com


class TriangleSelfAttention(nn.Module):
    def __init__(self):
        super(TriangleSelfAttention, self).__init__()

        self.dz = 64#model_args['Channel_z']
        self.dc = 8#model_args['Channel_c']
        self.num_head = 4#model_args['num_head']
        self.dhc = self.num_head * self.dc

        self.norm_com = nn.LayerNorm(self.dz)
        self.Linear_Q = nn.Linear(self.dz, self.dhc)
        self.Linear_K = nn.Linear(self.dz, self.dhc)
        self.Linear_V = nn.Linear(self.dz, self.dhc)
        #self.Linear_bias = nn.Linear(self.dz, self.num_head)

        self.softmax = nn.Softmax(-1)
        self.gate_v = nn.Linear(self.dz, self.dhc)
        self.Linear_final = nn.Linear(self.dhc, self.dz)


    def reshape_dim(self, x):
        new_shape = x.size()[:-1] + (self.num_head, self.dc)
        return x.view(*new_shape)

    def forward(self, z_com, mask=None, mask_sa=None, eps=5e4):

        B, row, col, _ = z_com.shape
        z_com = self.norm_com(z_com)

        scalar = torch.sqrt( torch.tensor(1.0/self.dc) )
        q = self.reshape_dim(self.Linear_Q(z_com))
        k = self.reshape_dim(self.Linear_K(z_com))
        v = self.reshape_dim(self.Linear_V(z_com))
        #bias = self.Linear_bias(z_com).permute(0,3,1,2)

        #coef = torch.exp(-(dist/8.0)**2.0/2.0).unsqueeze(2).type_as(q)

        attn = torch.einsum(f"bnihc, bnjhc->bhnij", q * scalar, k)
        if mask != None:
            attn = attn - ((1-mask[:,:,None,None,:])*eps).type_as(attn)
        if mask_sa != None:
            attn = attn - ((1-mask_sa[:,:,:,None,:])*eps).type_as(attn)
        #attn = attn * coef

        if attn.dtype is torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False):
            #attn_weights = self.softmax(attn)
                attn_weights = torch.nn.functional.softmax(attn, -1)
        else:
            attn_weights = torch.nn.functional.softmax(attn, -1)

        v_avg = torch.einsum(f"bhnij, bnjhc->bnihc",attn_weights, v)
        gate_v = (self.reshape_dim(self.gate_v(z_com))).sigmoid()
        z_com = (v_avg * gate_v).contiguous().view( v_avg.size()[:-2] + (-1,) )

        z_final = self.Linear_final(z_com)

        return  z_final

class Transition(nn.Module):

    def __init__(self):
        super(Transition, self).__init__()

        self.dz =  64#model_args['Channel_z']
        self.n = 4#model_args['Transition_n']

        self.norm = nn.LayerNorm(self.dz)
        self.transition = nn.Sequential(   nn.Linear(self.dz, self.dz*self.n),
                                           nn.ReLU(),
                                           nn.Linear(self.dz*self.n, self.dz)
                                        )
    def forward(self, z_com):

        z_com = self.norm(z_com)
        z_com = self.transition(z_com)

        return z_com



class EGNN_TriangularAttnNet(nn.Module):
    def __init__(self, prot, prot_egnn_hidden, prot_e, prot_egnn_layers,  rna, rna_egnn_hidden, rna_e, rna_egnn_layers, mid_c, out_c, device, attention):
        super().__init__()
        self.device = device
        self.midconv = nn.Conv2d(mid_c*2, 64, kernel_size = 1)
        self.outconv = nn.Conv2d(64, out_c, kernel_size = 1)
        self.egnn_prot = EGNN(prot, prot_egnn_hidden, mid_c, prot_e, device, nn.SiLU(), prot_egnn_layers)#, attention)
        self.egnn_rna = EGNN(rna, rna_egnn_hidden, mid_c, rna_e, device, nn.SiLU(), rna_egnn_layers)#, attention) 
        self.triangle_mul = TriangleMultiplication()
        self.triangle_attn = TriangleSelfAttention()
        self.transition = Transition()
        self.TriangleMulti = nn.ModuleList([ TriangleMultiplication() for _ in range(20) ])
        self.TriangleSelfR = nn.ModuleList([ TriangleSelfAttention() for _ in range(20) ])
        self.mid_c = mid_c
        self.in1 = nn.InstanceNorm2d(mid_c)
        self.in2 = nn.InstanceNorm2d(mid_c*2)
        self.norm_final = nn.LayerNorm(64)
        self.drop = nn.Dropout(0.10)
        self.resnet_com = Res_middle()

    def shortcut(self, input_, residual, is_first_block):
        if(is_first_block):
            in_shape = input_.shape[1]
            out_shape = residual.shape[1]
            input_ = self.conv_shortcut(input_)
        return torch.add(input_, residual)

    def normalization_block(self, x):
        norm1 = self.in1(x)
        norm2 = torch.nn.functional.normalize(x, p=2.0, dim=2, eps=1e-12) #row normalization
        norm3 = torch.nn.functional.normalize(x, p=2.0, dim=3, eps=1e-12) #column normalization
        x = torch.cat((norm1, norm2, norm3), dim=1)
        x = self.elu(x)
        return x    

    def forward(self, nodeFeats_prot, xyz_prot, edges_prot, edge_att_prot, nodeFeats_rna, xyz_rna, edges_rna, edge_att_rna):
        
        x, _ = self.egnn_prot(nodeFeats_prot, xyz_prot, edges_prot, edge_att_prot)#, self.device)
        y, _ = self.egnn_rna(nodeFeats_rna, xyz_rna, edges_rna, edge_att_rna)#, self.device)
        x = x.squeeze()#Pxoutdim
        y = y.squeeze()#Rxoutdim

        prot_ori_l = x.shape[0]
        rna_ori_l = y.shape[0]

        x = torch.permute(x, (1,0)).to(self.device)#to('cuda:2')
        y = torch.permute(y, (1,0)).to(self.device)#to('cuda:2')



        xy = torch.zeros(x.shape[0]+y.shape[0], x.shape[-1], y.shape[-1]).to(self.device)#to('cuda:2')
        for ii in range(x.shape[-1]):
            for jj in range(y.shape[-1]):
                xy[:, ii, jj] = torch.cat((x[:,ii], y[:, jj]),dim=0)

        xy = xy.unsqueeze(dim=0)
        x = self.in2(xy)
        x = self.midconv(x)


        x, _ = self.resnet_com(x)



        x = x.permute(0, 2, 3, 1)
        for idx in range(20):
            x = x + self.drop(self.TriangleSelfR[idx](x))
            x = x + self.transition(x)
        x = self.norm_final(x)
        x = x.permute(0, 3, 1, 2)
        x = self.outconv(x)

        return x


