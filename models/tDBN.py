import time
from enum import Enum
from functools import reduce

import numpy as np
import sparseconvnet as scn
import torch
from torch import nn
from torch.nn import functional as F

import torchplus
from torchplus import metrics
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
from tDBN.core import box_torch_ops
from tDBN.core.losses import (WeightedSigmoidClassificationLoss,
                              WeightedSmoothL1LocalizationLoss,
                              WeightedSoftmaxClassificationLoss)
import operator
import torch
import warnings
from torch.nn.parallel.data_parallel import *

class tDBN_1(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_filters_down1=[ 64, 128, 256, 512],
                 num_filters_down2=[ 64, 128, 256, 512],
                 name='tDBN_1'):
        super(tDBN_1, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        sparse_shape = np.array(output_shape[1:4]) # + [1, 0, 0]

        # print(sparse_shape)
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape
        middle_layers = []

        num_filter_fpn        = num_filters_down1 # [ 64, 128, 256, 512] #dimension of feature maps, num_filter_fpn[3] == dimension_feature_map[3]
        dimension_feature_map = num_filters_down2 # [ 64, 128, 256, 512] # dimensions of output into 2D feature map
        dimension_kernel_size = [15,  7,   3,  1 ]   #(input.spatial_size - self.filter_size) / self.filter_stride + 1

        # -----------------------------------------------------------------
        ## block1 and feature map 0, convert from voxel into 3D tensor
        # -----------------------------------------------------------------
        for i, o in [[128, num_filter_fpn[0]]]: #, [num_filter_fpn[0], num_filter_fpn[0]]]:
            middle_layers.append(scn.SubmanifoldConvolution(3, i, o, 3, False))
            middle_layers.append(scn.BatchNormReLU(o, eps=1e-3, momentum=0.99))

        self.block0 = Sequential(*middle_layers)
        middle_layers = []
        # downsampling with one 15*15 kernel, to compress 15 dimension into 1
        # dimension, 64*15*399*351 ---> 256*1*399*351
        middle_layers.append(
            scn.Convolution(
                3,
                num_filter_fpn[0],
                dimension_feature_map[0], (dimension_kernel_size[0], 1, 1), (2, 1, 1),#in:7, out:5
                bias=False))
        middle_layers.append(
            scn.BatchNormReLU( dimension_feature_map[0] , eps=1e-3, momentum=0.99))
        ## compressed into a birdview, height dimension into 1
        middle_layers.append(scn.SparseToDense(3, dimension_feature_map[0]))
        self.feature_map0 =   Sequential(*middle_layers)   ## feature map1
        middle_layers = []

        # --------------------------------------------------------------
        ## block1-3 and feature map1-3
        # --------------------------------------------------------------
        for k in range(1,4):
            ## reduce the resolution
            middle_layers.append(
                scn.Convolution(
                    3,
                    num_filter_fpn[k-1],
                    num_filter_fpn[k], (3, 3, 3), (2, 2, 2),
                    bias=False))
            middle_layers.append(
                scn.BatchNormReLU(num_filter_fpn[k], eps=1e-3, momentum=0.99))
            # 128*7*199*175 recurrent
            for i, o in [[num_filter_fpn[k], num_filter_fpn[k]], [num_filter_fpn[k], num_filter_fpn[k]]]:
                middle_layers.append(scn.SubmanifoldConvolution(3, i, o, 3, False))
                middle_layers.append(scn.BatchNormReLU(o, eps=1e-3, momentum=0.99))

            if   k==1:
                self.block1 = Sequential(*middle_layers)
            elif k==2:
                self.block2 = Sequential(*middle_layers)
            elif k==3:
                self.block3 = Sequential(*middle_layers)

            middle_layers = []

            ## compressed into a birdview, height dimension into 1
            middle_layers.append(
                scn.Convolution(
                    3,
                    num_filter_fpn[k],
                    dimension_feature_map[k], (dimension_kernel_size[k], 1, 1), (1, 1, 1),
                    bias=False)
            )
            middle_layers.append(
                scn.BatchNormReLU( dimension_feature_map[k], eps=1e-3, momentum=0.99))
            middle_layers.append(scn.SparseToDense(3, dimension_feature_map[k]))   ## convert a SparseConvNet hidden layer to a dense convolutional layer

            if   k==1 :
                self.feature_map1 = Sequential(*middle_layers) # in:7, out:5
            elif k==2:
                self.feature_map2 = Sequential(*middle_layers) # in:3, out:1
                # self.feature_map2 = Sequential(scn.SparseToDense(3, dimension_feature_map[k]))

            elif k==3:
                #self.feature_map3 = Sequential(*middle_layers)  # XXX
                self.feature_map3 = Sequential(scn.SparseToDense(3, dimension_feature_map[k]))  ## last one is the 2D instead of 3D

            middle_layers = []


    def forward(self, voxel_features, coors, batch_size):

        coors = coors.int()[:, [1, 2, 3, 0]]
        ret = self.scn_input((coors.cpu(), voxel_features, batch_size))
        output = {}
        for k in range(4):
            if   k==0:
                 ret = self.block0(ret)
            elif k==1:
                 ret = self.block1(ret)
            elif k==2:
                 ret = self.block2(ret)
            elif k==3:
                ret = self.block3(ret)

            temp = []

            if   k==0:
                temp = self.feature_map0(ret) # D: 5
            elif k==1:
                temp = self.feature_map1(ret) # D: 3
            elif k==2:
                temp = self.feature_map2(ret) # D: 2
            elif k==3:
                temp = self.feature_map3(ret)

            N, C, D, H, W = temp.shape
            output[k] = temp.view(N, C*D, H, W)

        return output


class tDBN_2(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_filters_down1=[ 32, 64, 96, 128],
                 num_filters_down2=[ 32, 64, 96, 128],
                 name='tDBN_2'):
        super(tDBN_2, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        sparse_shape = np.array(output_shape[1:4]) # + [1, 0, 0]
        # print(sparse_shape)
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape

        To_use_bias = False

        num_filter_fpn        = num_filters_down1 # [ 64, 128, 256, 512] #dimension of feature maps, num_filter_fpn[3] == dimension_feature_map[3]
        dimension_feature_map = num_filters_down2 # [ 64, 128, 256, 512] # dimensions of output into 2D feature map
        dimension_kernel_size = [ 15,   7,   3, 1  ]

        m = None
        m = scn.Sequential()
        # -----------------------------------------------------------------
        ## block1 and feature map 0, convert from voxel into 3D tensor
        # -----------------------------------------------------------------
        for i, o in [[128, num_filter_fpn[0]]]: # , [num_filter_fpn[0], num_filter_fpn[0]]]:
            m.add(scn.SubmanifoldConvolution(3, i, o, 3, False))

        self.block_input = m
        middle_layers = []

        reps = 2
        dimension = 3
        residual_use = True # using residual block or not
        leakiness = 0

        m = None
        m = scn.Sequential()
        for _ in range(reps):
            self.block(m, num_filter_fpn[0], num_filter_fpn[0], residual_blocks = residual_use)
        self.x0_in = m

        for k in range(1,4):
            m = None
            m = scn.Sequential()
            # downsample
            m.add(scn.BatchNormLeakyReLU(num_filter_fpn[k-1],leakiness=leakiness)).add(
              scn.Convolution(dimension, num_filter_fpn[k-1], num_filter_fpn[k], 3, 2, False))
            # cnn
            for _ in range(reps):
                if k == 4:
                    self.block(m, num_filter_fpn[k], num_filter_fpn[k], dimension=2, residual_blocks = residual_use)  ## it has be compressed into 2 dimensions
                else:
                    self.block(m, num_filter_fpn[k], num_filter_fpn[k], dimension=3, residual_blocks = residual_use)
            if k == 1:
                self.x1_in = m
            if k == 2:
                self.x2_in = m
            if k == 3:
                self.x3_in = m


        self.feature_map3 = scn.Sequential(scn.BatchNormLeakyReLU(num_filter_fpn[3],leakiness=leakiness)
                                       ).add(scn.SparseToDense(3, num_filter_fpn[3]))  ## last one is the 2D instead of 3D

        for k in range(2,-1,-1):
            m = None
            m = scn.Sequential()
            # upsample
            m.add(
                scn.BatchNormLeakyReLU(num_filter_fpn[k+1],leakiness=leakiness)).add(
                        scn.Deconvolution(dimension, num_filter_fpn[k+1], num_filter_fpn[k],
                            3, 2, False))
            if k == 2:
                self.upsample32 = m
            if k == 1:
                self.upsample21 = m
            if k == 0:
                self.upsample10 = m

            m = None
            m = scn.Sequential()
            m.add(scn.JoinTable())
            for i in range(reps):
                self.block(m, num_filter_fpn[k] * (2 if i == 0 else 1), num_filter_fpn[k], residual_blocks = residual_use)

            if k == 2:
                self.concate2 = m

            if k == 1:
                self.concate1 = m

            if k == 0:
                self.concate0 = m


            m = None
            m = scn.Sequential()
            m.add(  scn.BatchNormLeakyReLU(num_filter_fpn[k],leakiness=leakiness)).add(scn.Convolution(
                    3,
                    num_filter_fpn[k],
                    dimension_feature_map[k], (dimension_kernel_size[k], 1, 1), (1, 1, 1),
                    bias=False)).add(
                        scn.BatchNormReLU( dimension_feature_map[k], eps=1e-3, momentum=0.99)).add(
                                scn.SparseToDense(3, dimension_feature_map[k]))
            if k == 2:
                self.feature_map2 = m

            if k == 1:
                self.feature_map1 = m

            if k == 0:
                self.feature_map0 = m


    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()[:, [1, 2, 3, 0]]
        ret = self.scn_input((coors.cpu(), voxel_features, batch_size))

        output = {}
        ret = self.block_input(ret)

        x0 = self.x0_in(ret)
        x1 = self.x1_in(x0)
        x2 = self.x2_in(x1)
        x3 = self.x3_in(x2)

        x2_f = self.concate2([x2, self.upsample32(x3)])
        x1_f = self.concate1([x1, self.upsample21(x2_f)])
        x0_f = self.concate0([x0, self.upsample10(x1_f)])

        # generate output feature maps
        x0_out = self.feature_map0(x0_f)
        x1_out = self.feature_map1(x1_f)
        x2_out = self.feature_map2(x2_f)
        x3_out = self.feature_map3(x3)

        #
        N, C, D, H, W = x0_out.shape
        output[0] = x0_out.view(N, C*D, H, W)

        N, C, D, H, W = x1_out.shape
        output[1] = x1_out.view(N, C*D, H, W)

        N, C, D, H, W = x2_out.shape
        output[2] = x2_out.view(N, C*D, H, W)

        N, C, D, H, W = x3_out.shape
        output[3] = x3_out.view(N, C*D, H, W)

        return output


    def block(self, m, a, b, dimension=3, residual_blocks=False, leakiness=0):  # default using residual_block
        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))


class tDBN_bv_1(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_filters_down1=[ 32, 64, 96, 128],
                 num_filters_down2=[ 32, 64, 96, 128],
                 name='tDBN_bv_1'):
        super(tDBN_bv_1, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        sparse_shape = np.array(output_shape[1:4]) # + [1, 0, 0]
        # sparse_shape[0] = 11
        # print(sparse_shape)
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape

        To_use_bias = False
        residual_use = True # using residual block or not
        dimension = 3

        reps = 2
        dimension = 3
        leakiness = 0
        input_filters_layers  =  num_filters_down1[:4]  # feature channels in the raw data.


        num_filter_fpn        = num_filters_down1[3:] # [ 64, 128, 256, 512] #dimension of feature maps, num_filter_fpn[3] == dimension_feature_map[3]
        dimension_feature_map = num_filters_down2 # [ 64, 128, 256, 512] # dimensions of output into 2D feature map
        dimension_kernel_size = [ 15,   7,   3, 1  ]

        filters_input_pairs =  [[input_filters_layers[i], input_filters_layers[i + 1]]
                                  for i in range(len(input_filters_layers)-1)]

        m = None
        m = scn.Sequential()
        # -----------------------------------------------------------------
        ## block1 and feature map 0, convert from voxel into 3D tensor
        # -----------------------------------------------------------------
        for i, o in [[1, input_filters_layers[0]]]:
            m.add(scn.SubmanifoldConvolution(3, i, o, 3, False))

        for i, o in filters_input_pairs: # , [num_filter_fpn[0], num_filter_fpn[0]]]:
            for _ in range(reps):
                self.block(m ,i ,i ,residual_blocks = residual_use)

            m.add(scn.BatchNormLeakyReLU(i ,leakiness=leakiness)).add(
              scn.Convolution(dimension, i, o, 3, 2, False))


        self.block_input = m
        middle_layers = []


        m = None
        m = scn.Sequential()
        for _ in range(reps):
            self.block(m, num_filter_fpn[0], num_filter_fpn[0], residual_blocks = residual_use)
        self.block0 = m

        m = None
        m = scn.Sequential()
        m.add(scn.BatchNormLeakyReLU(num_filter_fpn[0],leakiness=leakiness)).add(
                scn.Convolution(3, num_filter_fpn[0], dimension_feature_map[0], (dimension_kernel_size[0], 1, 1), (1, 1, 1),bias=False)).add(
                scn.BatchNormLeakyReLU( dimension_feature_map[0],leakiness=leakiness)).add(
                scn.SparseToDense(3, dimension_feature_map[0]))
        self.feature_map0 = m



        for k in range(1,4):
            m = None
            m = scn.Sequential()
            # downsample
            m.add(scn.BatchNormLeakyReLU(num_filter_fpn[k-1],leakiness=leakiness)).add(
              scn.Convolution(dimension, num_filter_fpn[k-1], num_filter_fpn[k], 3, 2, False))
            # cnn
            for _ in range(reps):
                if k == 4:
                    self.block(m, num_filter_fpn[k], num_filter_fpn[k], dimension=2, residual_blocks = residual_use)  ## it has be compressed into 2 dimensions
                else:
                    self.block(m, num_filter_fpn[k], num_filter_fpn[k], dimension=3, residual_blocks = residual_use)
            if k == 1:
                self.block1 = m
            if k == 2:
                self.block2 = m
            if k == 3:
                self.block3 = m

            m = None
            m = scn.Sequential()
            m.add(scn.BatchNormLeakyReLU(num_filter_fpn[k],leakiness=leakiness)).add(
                scn.Convolution(3, num_filter_fpn[k], dimension_feature_map[k], (dimension_kernel_size[k], 1, 1), (1, 1, 1),bias=False))
            m.add(scn.BatchNormLeakyReLU( dimension_feature_map[k],leakiness=leakiness)).add(
                scn.SparseToDense(3, dimension_feature_map[k]))

            if k==1:
                self.feature_map1 = m
            elif k==2:
                self.feature_map2 = m
            elif k==3:
                self.feature_map3 = scn.Sequential(scn.BatchNormLeakyReLU(num_filter_fpn[3],leakiness=leakiness)
                                       ).add(scn.SparseToDense(3, num_filter_fpn[3]))  ## last one is the 2D instead of 3D


    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()[:, [1, 2, 3, 0]]
        ret = self.scn_input((coors.cpu(), voxel_features, batch_size))

        output = {}
        ret = self.block_input(ret)

        for k in range(4):
            if   k==0:
                 ret = self.block0(ret)
            elif k==1:
                 ret = self.block1(ret)
            elif k==2:
                 ret = self.block2(ret)
            elif k==3:
                ret = self.block3(ret)

            temp = []

            if   k==0:
                temp = self.feature_map0(ret) # D: 5
            elif k==1:
                temp = self.feature_map1(ret) # D: 3
            elif k==2:
                temp = self.feature_map2(ret) # D: 2
            elif k==3:
                temp = self.feature_map3(ret)

            N, C, D, H, W = temp.shape
            output[k] = temp.view(N, C*D, H, W)

        return output


    def block(self, m, a, b, dimension=3, residual_blocks=False, leakiness=0):  # default using residual_block
        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))

class tDBN_bv_2(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_filters_down1=[ 32, 64, 96, 128],
                 num_filters_down2=[ 32, 64, 96, 128],
                 name='tDBN_bv_2'):
        super(tDBN_bv_2, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        sparse_shape = np.array(output_shape[1:4]) # + [1, 0, 0]
        # sparse_shape[0] = 11
        # print(sparse_shape)
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape

        To_use_bias = False
        residual_use = True # using residual block or not
        dimension = 3

        reps = 2
        dimension = 3
        leakiness = 0
        input_filters_layers  =  num_filters_down1[:4]  # feature channels in the raw data.


        num_filter_fpn        = num_filters_down1[3:] # [ 64, 128, 256, 512] #dimension of feature maps, num_filter_fpn[3] == dimension_feature_map[3]
        dimension_feature_map = num_filters_down2 # [ 64, 128, 256, 512] # dimensions of output into 2D feature map
        dimension_kernel_size = [ 15,   7,   3, 1  ]

        filters_input_pairs =  [[input_filters_layers[i], input_filters_layers[i + 1]]
                                  for i in range(len(input_filters_layers)-1)]


        m = None
        m = scn.Sequential()
        # -----------------------------------------------------------------
        ## block1 and feature map 0, convert from voxel into 3D tensor
        # -----------------------------------------------------------------
        for i, o in [[1, input_filters_layers[0]]]:
            m.add(scn.SubmanifoldConvolution(3, i, o, 3, False))

        for i, o in filters_input_pairs: # , [num_filter_fpn[0], num_filter_fpn[0]]]:
            for _ in range(reps):
                self.block(m ,i ,i ,residual_blocks = residual_use)

            m.add(scn.BatchNormLeakyReLU(i ,leakiness=leakiness)).add(
              scn.Convolution(dimension, i, o, 3, 2, False))



        self.block_input = m
        middle_layers = []


        m = None
        m = scn.Sequential()
        for _ in range(reps):
            self.block(m, num_filter_fpn[0], num_filter_fpn[0], residual_blocks = residual_use)
        self.x0_in = m

        for k in range(1,4):
            m = None
            m = scn.Sequential()
            # downsample
            m.add(scn.BatchNormLeakyReLU(num_filter_fpn[k-1],leakiness=leakiness)).add(
              scn.Convolution(dimension, num_filter_fpn[k-1], num_filter_fpn[k], 3, 2, False))
            # cnn
            for _ in range(reps):
                if k == 4:
                    self.block(m, num_filter_fpn[k], num_filter_fpn[k], dimension=2, residual_blocks = residual_use)  ## it has be compressed into 2 dimensions
                else:
                    self.block(m, num_filter_fpn[k], num_filter_fpn[k], dimension=3, residual_blocks = residual_use)
            if k == 1:
                self.x1_in = m
            if k == 2:
                self.x2_in = m
            if k == 3:
                self.x3_in = m


        #self.feature_map3 = Sequential(*middle_layers)  # XXX
        self.feature_map3 = scn.Sequential(scn.BatchNormLeakyReLU(num_filter_fpn[3],leakiness=leakiness)
                                       ).add(scn.SparseToDense(3, num_filter_fpn[3]))  ## last one is the 2D instead of 3D

        for k in range(2,-1,-1):
            m = None
            m = scn.Sequential()
            # upsample
            m.add(
                scn.BatchNormLeakyReLU(num_filter_fpn[k+1],leakiness=leakiness)).add(
                        scn.Deconvolution(dimension, num_filter_fpn[k+1], num_filter_fpn[k],
                            3, 2, False))
            if k == 2:
                self.upsample32 = m
            if k == 1:
                self.upsample21 = m
            if k == 0:
                self.upsample10 = m

            m = None
            m = scn.Sequential()
            m.add(scn.JoinTable())
            for i in range(reps):
                self.block(m, num_filter_fpn[k] * (2 if i == 0 else 1), num_filter_fpn[k], residual_blocks = residual_use)

            if k == 2:
                self.concate2 = m

            if k == 1:
                self.concate1 = m

            if k == 0:
                self.concate0 = m


            m = None
            m = scn.Sequential()
            m.add(  scn.BatchNormLeakyReLU(num_filter_fpn[k],leakiness=leakiness)).add(scn.Convolution(
                    3,
                    num_filter_fpn[k],
                    dimension_feature_map[k], (dimension_kernel_size[k], 1, 1), (1, 1, 1),
                    bias=False)).add(
                        scn.BatchNormReLU( dimension_feature_map[k], eps=1e-3, momentum=0.99)).add(
                                scn.SparseToDense(3, dimension_feature_map[k]))
            if k == 2:
                self.feature_map2 = m

            if k == 1:
                self.feature_map1 = m

            if k == 0:
                self.feature_map0 = m



    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()[:, [1, 2, 3, 0]]
        ret = self.scn_input((coors.cpu(), voxel_features, batch_size))

        output = {}
        ret = self.block_input(ret)

        x0 = self.x0_in(ret)
        x1 = self.x1_in(x0)
        x2 = self.x2_in(x1)
        x3 = self.x3_in(x2)

        x2_f = self.concate2([x2, self.upsample32(x3)])
        x1_f = self.concate1([x1, self.upsample21(x2_f)])
        x0_f = self.concate0([x0, self.upsample10(x1_f)])

        # generate output feature maps
        x0_out = self.feature_map0(x0_f)
        x1_out = self.feature_map1(x1_f)
        x2_out = self.feature_map2(x2_f)
        x3_out = self.feature_map3(x3)

        #
        N, C, D, H, W = x0_out.shape
        output[0] = x0_out.view(N, C*D, H, W)

        N, C, D, H, W = x1_out.shape
        output[1] = x1_out.view(N, C*D, H, W)

        N, C, D, H, W = x2_out.shape
        output[2] = x2_out.view(N, C*D, H, W)

        N, C, D, H, W = x3_out.shape
        output[3] = x3_out.view(N, C*D, H, W)

        return output


    def block(self, m, a, b, dimension=3, residual_blocks=False, leakiness=0):  # default using residual_block
        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))

