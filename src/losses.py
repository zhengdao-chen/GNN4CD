import numpy as np
import math
import os
# import dependencies
import time

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

criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor

def compute_loss(pred, labels):
    pred = pred.view(-1, pred.size()[-1])
    labels = labels.view(-1)
    return criterion(pred, labels)

def compute_cosine_loss_bcd(pred, labels):
    # pred = pred.view(-1, pred.size()[-1])
    # labels = labels.view(-1)
    if (pred.data.shape[0] == 1):
        print (pred)
        print (labels)
        pred = pred.squeeze(0)
        labels_01_1 = ((labels + 1) / 2).squeeze(0)
        labels_01_2 = ((-1 * labels + 1) / 2).squeeze(0)
        print (labels_01_1)
        # print (labels_01_1.type(dtype_l))
        # print (labels_01_1.type(dtype_l).shape)
        loss = pred.dot(labels_01_1.type(dtype)).pow(2) * (-1)
    else:
        raise ValueError('batch size greater than 1')
    return loss

def compute_loss_bcd(pred, labels):
    # pred = pred.view(-1, pred.size()[-1])
    # labels = labels.view(-1)
    if (pred.data.shape[0] == 1):
        pred = pred.squeeze(0)
        labels_01_1 = ((labels + 1) / 2).squeeze(0)
        labels_01_2 = ((-1 * labels + 1) / 2).squeeze(0)
        # print (labels_01_1.type(dtype_l))
        # print (labels_01_1.type(dtype_l).shape)
        loss1 = criterion(pred, labels_01_1.type(dtype_l))
        loss2 = criterion(pred, labels_01_2.type(dtype_l))
        loss = torch.min(loss1, loss2)
    else:
        # print ('pred', pred)
        # print ('labels' labels)
        loss = 0
        for i in range(pred.data.shape[0]):
            pred_single = pred[i, :, :]
            labels_single = labels[i, :]
            # pred = pred.squeeze(0)
            labels_01_1 = ((labels_single + 1) / 2)#.squeeze(0)
            labels_01_2 = ((-1 * labels_single + 1) / 2)#.squeeze(0)
            # print (labels_01_1.type(dtype_l))
            # print (labels_01_1.type(dtype_l).shape)
            loss1 = criterion(pred_single, labels_01_1.type(dtype_l))
            loss2 = criterion(pred_single, labels_01_2.type(dtype_l))
            loss_single = torch.min(loss1, loss2)
            loss += loss_single
    return loss

def from_scores_to_labels_mcd_batch(pred):
    # pred = pred.squeeze(0)
    # labels_pred = (pred[:, :, 0] > pred[:, :, 1]).type(dtype) * 2.0 - 1
    labels_pred = np.argmax(pred, axis = 2).astype(int)
    return labels_pred

def compute_accuracy_mcd_batch(labels_pred, labels):
    # print ('pred', labels_pred)
    # print ('labels', labels)
    overlap = (labels_pred == labels).astype(int)
    acc = np.mean(labels_pred == labels)
    return acc

def compute_loss_multiclass(pred_llh, labels, n_classes):
    loss = 0
    permutations = permuteposs(n_classes)
    if (torch.cuda.is_available()):
        batch_size = pred_llh.data.cpu().shape[0]
    else:
        batch_size = pred_llh.data.shape[0]
    for i in range(batch_size):
        pred_llh_single = pred_llh[i, :, :]
        labels_single = labels[i, :]
        for j in range(permutations.shape[0]):
            permutation = permutations[j, :]
            # print (labels_single.data.numpy())
            if (torch.cuda.is_available()):
                labels_under_perm = torch.from_numpy(permutations[j, labels_single.data.cpu().numpy().astype(int)])
            else:
                labels_under_perm = torch.from_numpy(permutations[j, labels_single.data.numpy().astype(int)])

            # print ('pred', pred_llh_single)
            # print ('labels', Variable(labels_under_perm.type(dtype_l), volatile=False))

            loss_under_perm = criterion(pred_llh_single, Variable(labels_under_perm.type(dtype_l), volatile=False))

            if (j == 0):
                loss_single = loss_under_perm
            else:
                loss_single = torch.min(loss_single, loss_under_perm)

        loss += loss_single
    return loss

def compute_accuracy_multiclass(pred_llh, labels, n_classes):
    if (torch.cuda.is_available()):
        pred_llh = pred_llh.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
    else:
        pred_llh = pred_llh.data.numpy()
        labels = labels.data.numpy()
    batch_size = pred_llh.shape[0]
    pred_labels = from_scores_to_labels_mcd_batch(pred_llh)
    # print ('predicted labels', np.transpose(labels_pred))
    # accuracy = compute_accuracy_mcd_batch(labels_pred, labels)
    # self.accuracy_test.append(accuracy)
    acc = 0
    permutations = permuteposs(n_classes)
    for i in range(batch_size):
        pred_labels_single = pred_labels[i, :]
        labels_single = labels[i, :]
        for j in range(permutations.shape[0]):
            permutation = permutations[j, :]
            # print (labels_single.data.numpy())
            labels_under_perm = permutations[j, labels_single.astype(int)]

            # print ('pred_labels', pred_labels_single)
            # print ('labels', labels_under_perm)

            acc_under_perm = compute_accuracy_mcd_batch(pred_labels_single, labels_under_perm)
            # print ('i', i, 'acc', acc_under_perm)
            if (j == 0):
                acc_single = acc_under_perm
            else:
                acc_single = np.max([acc_single, acc_under_perm])

        acc += acc_single
    acc = acc / labels.shape[0]
    acc = (acc - 1 / n_classes) / (1 - 1 / n_classes)
    return acc

def permuteposs(n_classes):
    permutor = Permutor(n_classes)
    permutations = permutor.return_permutations()
    # print(permutations)
    return permutations


class Permutor:
    def __init__(self, n_classes):
        self.row = 0
        self.n_classes = n_classes
        self.collection = np.zeros([math.factorial(n_classes), n_classes])

    def permute(self, arr, l, r): 
        if l==r: 
            self.collection[self.row, :] = arr
            self.row += 1
        else: 
            for i in range(l,r+1): 
                arr[l], arr[i] = arr[i], arr[l] 
                self.permute(arr, l+1, r) 
                arr[l], arr[i] = arr[i], arr[l]

    def return_permutations(self):
        self.permute(np.arange(self.n_classes), 0, self.n_classes-1)
        return self.collection
                

if __name__ == '__main__':
    pred = Variable(torch.randn(2, 3, 5), volatile=False)
    labels_npy = np.zeros([2, 3])
    labels_npy[0, 0] = 1
    labels_npy[0, 1] = 2
    labels_npy[0, 2] = 0
    labels_npy[1, 0] = 4
    labels_npy[1, 1] = 0
    labels_npy[1, 2] = 2
    labels = Variable(torch.from_numpy(labels_npy), volatile=False)
