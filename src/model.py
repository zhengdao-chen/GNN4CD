#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib
matplotlib.use('Agg')

# Pytorch requirements
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor

def sinkhorn_knopp(A, iterations=1):
    A_size = A.size()
    for it in range(iterations):
        A = A.view(A_size[0]*A_size[1], A_size[2])
        A = F.softmax(A)
        A = A.view(*A_size).permute(0, 2, 1)
        A = A.view(A_size[0]*A_size[1], A_size[2])
        A = F.softmax(A)
        A = A.view(*A_size).permute(0, 2, 1)
    return A

def gmul(input):
    W, x = input
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]
    J = W_size[-1]
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output

def GMul(W, x):
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    # print (x)
    W_size = W.size()
    # print (W)
    N = W_size[-3]
    J = W_size[-1]
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output

class Gconv_last(nn.Module):
    def __init__(self, feature_maps, J):
        super(Gconv_last, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_outputs = feature_maps[2]
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, input):
        W = input[0]
        x = gmul(input) # out has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(x_size[0]*x_size[1], -1)
        x = self.fc(x) # has size (bs*N, num_outputs)
        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x

class Gconv_last_bcd(nn.Module):
    def __init__(self, feature_maps, J, n_classes):
        super(Gconv_last_bcd, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_outputs = n_classes
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, input):
        W = input[0]
        x = gmul(input) # out has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(x_size[0]*x_size[1], -1)
        x = self.fc(x) # has size (bs*N, num_outputs)
        # x = F.tanh(x) # added for last layer
        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x

class Gconv(nn.Module):
    def __init__(self, feature_maps, J):
        super(Gconv, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_outputs = feature_maps[2]
        self.fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.fc2 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):
        W = input[0]
        # print ('W size', W.size())
        # print ('x size', input[1].size())
        x = gmul(input) # out has size (bs, N, num_inputs)
        x_size = x.size()
        # print (x_size)
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        # print (x.size())
        x1 = F.relu(self.fc1(x)) # has size (bs*N, num_outputs)
        x2 = self.fc2(x)
        x = torch.cat((x1, x2), 1)
        x = self.bn(x)
        # print (x.size())
        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x

class Gconv_new(nn.Module):
    def __init__(self, feature_maps, J):
        super(Gconv_new, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_outputs = feature_maps[2]
        self.fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.fc2 = nn.Linear(self.num_inputs, self.num_outputs - self.num_outputs // 2)
        self.bn2d = nn.BatchNorm2d(self.num_outputs)

    def forward(self, input):
        W = input[0]
        # print ('W size', W.size())
        # print ('x size', input[1].size())
        x = gmul(input) # out has size (bs, N, num_inputs)
        x_size = x.size()
        # print (x_size)
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        # print (x.size())
        x1 = F.relu(self.fc1(x)) # has size (bs*N, num_outputs)
        x2 = self.fc2(x)
        x = torch.cat((x1, x2), 1)
        # x = self.bn2d(x.unsqueeze(0).unsqueeze(3)).squeeze(3).squeeze(0)
        x = self.bn2d(x)
        # print (x.size())
        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x

class gnn_atomic(nn.Module):
    def __init__(self, feature_maps, J):
        super(gnn_atomic, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_outputs = feature_maps[2]
        self.fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.fc2 = nn.Linear(self.num_inputs, self.num_outputs - self.num_outputs // 2)
        self.bn2d = nn.BatchNorm2d(self.num_outputs)

    def forward(self, WW, x):
        # W = input[0]
        # # print ('W size', W.size())
        # # print ('x size', input[1].size())
        # x = gmul(input) # out has size (bs, N, num_inputs)
        x = GMul(WW, x)
        x_size = x.size()
        # print (x_size)
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        # print (x.size())
        x1 = F.relu(self.fc1(x)) # has size (bs*N, num_outputs)
        x2 = self.fc2(x)
        x = torch.cat((x1, x2), 1)
        # x = self.bn2d(x.unsqueeze(0).unsqueeze(3)).squeeze(3).squeeze(0)
        x = self.bn2d(x)
        # print (x.size())
        x = x.view(*x_size[:-1], self.num_outputs)
        return WW, x

class gnn_atomic_final(nn.Module):
    def __init__(self, feature_maps, J, n_classes):
        super(gnn_atomic_final, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_outputs = n_classes
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, WW, x):
        x = GMul(WW, x) # out has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(x_size[0]*x_size[1], -1)
        x = self.fc(x) # has size (bs*N, num_outputs)
        # x = F.tanh(x) # added for last layer
        x = x.view(*x_size[:-1], self.num_outputs)
        return WW, x



class gnn_atomic_lg(nn.Module):
    def __init__(self, feature_maps, J):
        super(gnn_atomic_lg, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_inputs_2 = 2 * feature_maps[1]
        # self.num_inputs_3 = 4 * feature_maps[2]
        self.num_outputs = feature_maps[2]
        self.fcx2x_1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.fcy2x_1 = nn.Linear(self.num_inputs_2, self.num_outputs // 2)
        self.fcx2x_2 = nn.Linear(self.num_inputs, self.num_outputs - self.num_outputs // 2)
        self.fcy2x_2 = nn.Linear(self.num_inputs_2, self.num_outputs - self.num_outputs // 2)
        self.fcx2y_1 = nn.Linear(self.num_inputs_2, self.num_outputs // 2)
        self.fcy2y_1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.fcx2y_2 = nn.Linear(self.num_inputs_2, self.num_outputs - self.num_outputs // 2)
        self.fcy2y_2 = nn.Linear(self.num_inputs, self.num_outputs - self.num_outputs // 2)
        self.bn2d_x = nn.BatchNorm2d(self.num_outputs)
        self.bn2d_y = nn.BatchNorm2d(self.num_outputs)

    def forward(self, WW, x, WW_lg, y, P):
        # print ('W size', W.size())
        # print ('x size', input[1].size())
        x2x = GMul(WW, x) # out has size (bs, N, num_inputs)
        x2x_size = x2x.size()
        # print (x_size)
        x2x = x2x.contiguous()
        x2x = x2x.view(-1, self.num_inputs)
        # print (x.size()) 
        # print ('x2x', x2x)
        x2x = x2x.type(dtype)

        # y2x = torch.bmm(P, y)
        y2x = GMul(P, y)
        y2x_size = y2x.size()
        y2x = y2x.contiguous()
        y2x = y2x.view(-1, self.num_inputs_2)

        y2x = y2x.type(dtype)

        # xy2x = x2x + y2x 
        xy2x = F.relu(self.fcx2x_1(x2x) + self.fcy2x_1(y2x)) # has size (bs*N, num_outputs)

        xy2x_l = self.fcx2x_2(x2x) + self.fcy2x_2(y2x)
        x_cat = torch.cat((xy2x, xy2x_l), 1)
        # x_output = self.bn2d_x(x_cat)
        x_output = self.bn2d_x(x_cat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)

        x_output = x_output.view(*x2x_size[:-1], self.num_outputs)


        y2y = GMul(WW_lg, y)
        y2y_size = y2y.size()
        y2y = y2y.contiguous()
        y2y = y2y.view(-1, self.num_inputs)

        y2y = y2y.type(dtype)

        # x2y = torch.bmm(torch.t(P), x)
        x2y = GMul(torch.transpose(P, 2, 1), x)
        x2y_size = x2y.size()
        x2y = x2y.contiguous()
        x2y = x2y.view(-1, self.num_inputs_2)

        x2y = x2y.type(dtype)

        # xy2y = x2y + y2y
        xy2y = F.relu(self.fcx2y_1(x2y) + self.fcy2y_1(y2y))

        xy2y_l = self.fcx2y_2(x2y) + self.fcy2y_2(y2y)

        y_cat = torch.cat((xy2y, xy2y_l), 1)
        # y_output = self.bn2d_x(y_cat)
        y_output = self.bn2d_y(y_cat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)

        y_output = y_output.view(*y2y_size[:-1], self.num_outputs)

        # WW = WW.type(dtype)

        return WW, x_output, WW_lg, y_output, P

class gnn_atomic_lg_final(nn.Module):
    def __init__(self, feature_maps, J, n_classes):
        super(gnn_atomic_lg_final, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_inputs_2 = 2 * feature_maps[1]
        self.num_outputs = n_classes
        self.fcx2x_1 = nn.Linear(self.num_inputs, self.num_outputs)
        self.fcy2x_1 = nn.Linear(self.num_inputs_2, self.num_outputs)

    def forward(self, W, x, W_lg, y, P):
        # print ('W size', W.size())
        # print ('x size', input[1].size())
        x2x = GMul(W, x) # out has size (bs, N, num_inputs)
        x2x_size = x2x.size()
        # print (x_size)
        x2x = x2x.contiguous()
        x2x = x2x.view(-1, self.num_inputs)
        # print (x.size()) 

        # y2x = torch.bmm(P, y)
        y2x = GMul(P, y)
        y2x_size = x2x.size()
        y2x = y2x.contiguous()
        y2x = y2x.view(-1, self.num_inputs_2)

        # xy2x = x2x + y2x 
        xy2x = self.fcx2x_1(x2x) + self.fcy2x_1(y2x) # has size (bs*N, num_outputs)

        x_output = xy2x.view(*x2x_size[:-1], self.num_outputs)

        return W, x_output

class GNN(nn.Module):
    def __init__(self, num_features, num_layers, J):
        super(GNN, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_in = [1, 1, num_features]
        self.featuremap_mi = [num_features, num_features, num_features]
        self.featuremap_end = [num_features, num_features, num_features]
        self.layer0 = Gconv(self.featuremap_in, J)
        for i in range(num_layers):
            module = Gconv(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = Gconv_last(self.featuremap_end, J)

    def forward(self, input):
        cur = self.layer0(input)
        for i in range(self.num_layers):
            cur = self._modules['layer{}'.format(i+1)](cur)
        out = self.layerlast(cur)
        return out[1]

class lGNN_multiclass(nn.Module):
    def __init__(self, num_features, num_layers, J, n_classes=2):
        super(lGNN_multiclass, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_in = [1, 1, num_features]
        self.featuremap_mi = [num_features, num_features, num_features]
        self.featuremap_end = [num_features, num_features, num_features]
        # self.layer0 = Gconv(self.featuremap_in, J)
        self.layer0 = gnn_atomic_lg(self.featuremap_in, J)
        for i in range(num_layers):
            # module = Gconv(self.featuremap_mi, J)
            module = gnn_atomic_lg(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = gnn_atomic_lg_final(self.featuremap_end, J, n_classes)

    def forward(self, W, x, W_lg, y, P):
        cur = self.layer0(W, x, W_lg, y, P)
        for i in range(self.num_layers):
            cur = self._modules['layer{}'.format(i+1)](*cur)
        out = self.layerlast(*cur)
        return out[1]


class GNN_bcd(nn.Module):
    def __init__(self, num_features, num_layers, J, n_classes=2):
        super(GNN_bcd, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_in = [1, 1, num_features]
        self.featuremap_mi = [num_features, num_features, num_features]
        self.featuremap_end = [num_features, num_features, num_features]
        # self.layer0 = Gconv(self.featuremap_in, J)
        self.layer0 = Gconv_new(self.featuremap_in, J)
        for i in range(num_layers):
            # module = Gconv(self.featuremap_mi, J)
            module = Gconv_new(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = Gconv_last_bcd(self.featuremap_end, J, n_classes)

    def forward(self, input):
        cur = self.layer0(input)
        for i in range(self.num_layers):
            cur = self._modules['layer{}'.format(i+1)](cur)
        out = self.layerlast(cur)
        return out[1]

class GNN_multiclass(nn.Module):
    def __init__(self, num_features, num_layers, J, n_classes=2):
        super(GNN_multiclass, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_in = [1, 1, num_features]
        self.featuremap_mi = [num_features, num_features, num_features]
        self.featuremap_end = [num_features, num_features, num_features]
        # self.layer0 = Gconv(self.featuremap_in, J)
        self.layer0 = gnn_atomic(self.featuremap_in, J)
        for i in range(num_layers):
            # module = Gconv(self.featuremap_mi, J)
            module = gnn_atomic(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = gnn_atomic_final(self.featuremap_end, J, n_classes)

    def forward(self, W, x):
        cur = self.layer0(W, x)
        for i in range(self.num_layers):
            cur = self._modules['layer{}'.format(i+1)](*cur)
        out = self.layerlast(*cur)
        return out[1]

if __name__ == '__main__':
    # test modules
    bs =  4
    num_features = 10
    num_layers = 5
    N = 8
    x = torch.ones((bs, N, num_features))
    W1 = torch.eye(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    W2 = torch.ones(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    J = 2
    W = torch.cat((W1, W2), 3)
    input = [Variable(W), Variable(x)]
    ######################### test gmul ##############################
    # feature_maps = [num_features, num_features, num_features]
    # out = gmul(input)
    # print(out[0, :, num_features:])
    ######################### test gconv ##############################
    # feature_maps = [num_features, num_features, num_features]
    # gconv = Gconv(feature_maps, J)
    # _, out = gconv(input)
    # print(out.size())
    ######################### test gnn ##############################
    # x = torch.ones((bs, N, 1))
    # input = [Variable(W), Variable(x)]
    # gnn = GNN(num_features, num_layers, J)
    # out = gnn(input)
    # print(out.size())
    ######################### test siamese gnn ##############################
    x = torch.ones((bs, N, 1))
    input1 = [Variable(W), Variable(x)]
    input2 = [Variable(W.clone()), Variable(x.clone())]
    siamese_gnn = Siamese_GNN(num_features, num_layers, J)
    out = siamese_gnn(input1, input2)
    print(out.size())
    print(out)

    gnn = GNN_bcd(num_features, num_layers, 2)
    out=gnn(input1)
    print(out.size())
    print(out)
