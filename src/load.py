#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import networkx

#Pytorch requirements
import unicodedata
import string
import re
import random
import argparse

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

if (torch.cuda.is_available()):
    dtype_sp = torch.cuda.sparse.FloatTensor
    dtype = torch.cuda.FloatTensor
else:
    dtype_sp = torch.sparse.FloatTensor
    dtype = torch.FloatTensor

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def compute_operators(W, J):
    N = W.shape[0]
    # print ('W', W)
    # print ('W size', W.size())
    # operators: {Id, W, W^2, ..., W^{J-1}, D, U}
    d = W.sum(1)
    D = np.diag(d)
    QQ = W.copy()
    WW = np.zeros([N, N, J + 2])
    WW[:, :, 0] = np.eye(N)
    for j in range(J):
        WW[:, :, j + 1] = QQ.copy()
        # QQ = np.dot(QQ, QQ)
        QQ = np.minimum(np.dot(QQ, QQ), np.ones(QQ.shape))
    WW[:, :, J + 1] = D
    # WW[:, :, J + 1] = np.ones((N, N)) * 1.0 / float(N)
    WW = np.reshape(WW, [N, N, J + 2])
    x = np.reshape(d, [N, 1])
    return WW, x

def compute_operators_sp(W, J):
    N = W.shape[0]
    # print ('W', W)
    # print ('W size', W.size())
    # operators: {Id, W, W^2, ..., W^{J-1}, D, U}
    d = W.sum(1)
    D = np.diag(d)
    QQ = W.copy()
    WW = []
    I = np.eye(N)
    I = torch.from_numpy(I)#.unsqueeze(0)
    I = to_sparse(I).type(dtype_sp)
    I = Variable(I, volatile=False)
    WW.append(I)
    for j in range(J):
        # QQc = np.expand_dims(QQ.copy(), 0)
        QQc = QQ.copy()
        QQc = torch.from_numpy(QQc)
        QQc = to_sparse(QQc).type(dtype_sp)
        QQc = Variable(QQc, volatile=False)
        WW.append(QQc)
        # QQ = np.dot(QQ, QQ)
        QQ = np.minimum(np.dot(QQ, QQ), np.ones(QQ.shape))
    # D = torch.from_numpy(np.expand_dims(D, 0))
    D = torch.from_numpy(D)
    D = to_sparse(D).type(dtype_sp)
    D = Variable(D, volatile=False)
    WW.append(D)
    # WW[:, :, J + 1] = np.ones((N, N)) * 1.0 / float(N)
    x = np.reshape(d, [N, 1])
    return WW, x

def compute_operators_noD(W, J):
    N = W.shape[0]
    # print ('W', W)
    # print ('W size', W.size())
    # operators: {Id, W, W^2, ..., W^{J-1}, D, U}
    d = W.sum(1)
    D = np.diag(d)
    QQ = W.copy()
    WW = np.zeros([N, N, J + 1])
    WW[:, :, 0] = np.eye(N)
    for j in range(J):
        WW[:, :, j + 1] = QQ.copy()
        # QQ = np.dot(QQ, QQ)
        QQ = np.minimum(np.dot(QQ, QQ), np.ones(QQ.shape))
    # WW[:, :, J] = np.ones((N, N)) * 1.0 / float(N)
    WW = np.reshape(WW, [N, N, J + 1])
    x = np.reshape(d, [N, 1])
    return WW, x

def get_Pm(W):
    N = W.shape[0]
    W = W * (np.ones([N, N]) - np.eye(N))
    M = int(W.sum()) // 2
    p = 0
    Pm = np.zeros([N, M * 2])
    for n in range(N):
        for m in range(n+1, N):
            if (W[n][m]==1):
                Pm[n][p] = 1
                Pm[m][p] = 1
                Pm[n][p + M] = 1
                Pm[m][p + M] = 1
                p += 1
    return Pm

def get_Pd(W):
    N = W.shape[0]
    W = W * (np.ones([N, N]) - np.eye(N))
    M = int(W.sum()) // 2
    p = 0
    Pd = np.zeros([N, M * 2])
    for n in range(N):
        for m in range(n+1, N):
            if (W[n][m]==1):
                Pd[n][p] = 1
                Pd[m][p] = -1
                Pd[n][p + M] = -1
                Pd[m][p + M] = 1
                p += 1
    return Pd


def get_P(W):
    P = np.concatenate((np.expand_dims(get_Pm(W), 2), np.expand_dims(get_Pd(W), 2)), axis=2)
    return P

def get_P_sp(W):
    # P = np.concatenate((np.expand_dims(get_Pm(W), 2), np.expand_dims(get_Pd(W), 2)), axis=2)
    P = []
    # Pm = np.expand_dims(get_Pm(W), 0)
    Pm = get_Pm(W)
    Pm = torch.from_numpy(Pm)
    Pm = to_sparse(Pm).type(dtype_sp)
    Pm = Variable(Pm, volatile=False)
    # Pd = np.expand_dims(get_Pd(W), 0)
    Pd = get_Pd(W)
    Pd = torch.from_numpy(Pd)
    Pd = to_sparse(Pd).type(dtype_sp)
    Pd = Variable(Pd, volatile=False)
    P.append(Pm)
    P.append(Pd)
    return P

def get_W_lg(W):
    W_lg = np.transpose(get_Pm(W)).dot(get_Pd(W))
    return W_lg

def get_NB(W):
    product = np.transpose(get_Pm(W)).dot(get_Pd(W))
    M_doubled = product.shape[0]
    NB = product * (product > 0)
    for i in range(M_doubled):
        NB[i, i] = 0
    return NB

def get_NB_2(W):
    Pm = get_Pm(W)
    Pd = get_Pd(W)
    Pf = (Pm + Pd) / 2
    Pt = (Pm - Pd) / 2
    NB = np.transpose(Pt).dot(Pf) * (1 - np.transpose(Pf).dot(Pt))
    return NB

# def get_NB_direct(W):
#     Pd = get_Pd(W)
#     M_doubled = Pd.shape[1]
#     NB = np.zeros(M_doubled, M_doubled)
#     for i in range(M_doubled):


def get_lg_inputs(W, J):
    if (W.ndim == 3):
        W = W[0, :, :]
    WW, x = compute_operators(W, J)
    # W_lg = get_W_lg(W)
    # W_lg = get_NB(W)
    W_lg = get_NB_2(W)
    WW_lg, y = compute_operators(W_lg, J)
    P = get_P(W)
    x = x.astype(float)
    y = y.astype(float)
    WW = WW.astype(float)
    WW_lg = WW_lg.astype(float)
    P = P.astype(float)
    WW = Variable(torch.from_numpy(WW).unsqueeze(0), volatile=False)
    x = Variable(torch.from_numpy(x).unsqueeze(0), volatile=False)
    WW_lg = Variable(torch.from_numpy(WW_lg).unsqueeze(0), volatile=False)
    y = Variable(torch.from_numpy(y).unsqueeze(0), volatile=False)
    P = Variable(torch.from_numpy(P).unsqueeze(0), volatile=False)
    return WW, x, WW_lg, y, P

def get_lg_inputs_noD(W, J):
    if (W.ndim == 3):
        W = W[0, :, :]
    WW, x = compute_operators_noD(W, J)
    # W_lg = get_W_lg(W)
    W_lg = get_NB_2(W)
    WW_lg, y = compute_operators_noD(W_lg, J)
    P = get_P(W)
    x = x.astype(float)
    y = y.astype(float)
    WW = WW.astype(float)
    WW_lg = WW_lg.astype(float)
    P = P.astype(float)
    WW = Variable(torch.from_numpy(WW).unsqueeze(0), volatile=False)
    x = Variable(torch.from_numpy(x).unsqueeze(0), volatile=False)
    WW_lg = Variable(torch.from_numpy(WW_lg).unsqueeze(0), volatile=False)
    y = Variable(torch.from_numpy(y).unsqueeze(0), volatile=False)
    P = Variable(torch.from_numpy(P).unsqueeze(0), volatile=False)
    return WW, x, WW_lg, y, P

def get_splg_inputs(W, J):
    if (W.ndim == 3):
        W = W[0, :, :]
    WW, x = compute_operators_sp(W, J)
    # W_lg = get_W_lg(W)
    W_lg = get_NB_2(W)
    WW_lg, y = compute_operators_sp(W_lg, J)
    P = get_P_sp(W)
    # x = x.astype(float)
    # y = y.astype(float)
    # WW = WW.astype(float)
    # WW_lg = WW_lg.astype(float)
    # P = P.astype(float)
    # WW = Variable(to_sparse(torch.from_numpy(WW).unsqueeze(0)), volatile=False)
    x = Variable(torch.from_numpy(x), volatile=False).type(dtype)
    # WW_lg = Variable(to_sparse(torch.from_numpy(WW_lg).unsqueeze(0)), volatile=False)
    y = Variable(torch.from_numpy(y), volatile=False).type(dtype)
    # P = Variable(to_sparse(torch.from_numpy(P).unsqueeze(0)), volatile=False)
    return WW, x, WW_lg, y, P

def get_gnn_inputs(W, J):
    W = W[0, :, :]
    WW, x = compute_operators(W, J)
    WW = WW.astype(float)
    WW = Variable(torch.from_numpy(WW).unsqueeze(0), volatile=False)
    x = Variable(torch.from_numpy(x).unsqueeze(0), volatile=False)
    return WW, x

def get_gnn_inputs_noD(W, J):
    W = W[0, :, :]
    WW, x = compute_operators_noD(W, J)
    WW = WW.astype(float)
    WW = Variable(torch.from_numpy(WW).unsqueeze(0), volatile=False)
    x = Variable(torch.from_numpy(x).unsqueeze(0), volatile=False)
    return WW, x


