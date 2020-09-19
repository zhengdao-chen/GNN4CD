import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


def compute_operators(W, J):
    N = W.shape[0]
    d = W.sum(1)
    D = np.diag(d)
    QQ = W.copy()
    WW = np.zeros([N, N, J + 2])
    WW[:, :, 0] = np.eye(N)
    for j in range(J):
        WW[:, :, j + 1] = QQ.copy()
        QQ = np.minimum(np.dot(QQ, QQ), np.ones(QQ.shape))
    WW[:, :, J + 1] = D
    WW = np.reshape(WW, [N, N, J + 2])
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
    WW = torch.tensor(WW, requires_grad=True).unsqueeze(0)
    x = torch.tensor(x, requires_grad=True).unsqueeze(0)
    WW_lg = torch.tensor(WW_lg, requires_grad=True).unsqueeze(0)
    y = torch.tensor(y, requires_grad=True).unsqueeze(0)
    P = torch.tensor(P, requires_grad=True).unsqueeze(0)
    return WW, x, WW_lg, y, P

def get_gnn_inputs(W, J):
    W = W[0, :, :]
    WW, x = compute_operators(W, J)
    WW = WW.astype(float)
    WW = torch.tensor(WW, requires_grad=True).unsqueeze(0)
    x = torch.tensor(x, requires_grad=True).unsqueeze(0)
    return WW, x



