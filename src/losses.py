import numpy as np
import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor


def from_scores_to_labels_multiclass_batch(pred):
    labels_pred = np.argmax(pred, axis = 2).astype(int)
    return labels_pred

def compute_accuracy_multiclass_batch(labels_pred, labels):
    overlap = (labels_pred == labels).astype(int)
    acc = np.mean(labels_pred == labels)
    return acc

def compute_loss_multiclass(pred_llh, labels, n_classes):
    loss = 0
    permutations = permuteposs(n_classes)
    batch_size = pred_llh.data.cpu().shape[0]
    for i in range(batch_size):
        pred_llh_single = pred_llh[i, :, :]
        labels_single = labels[i, :]
        for j in range(permutations.shape[0]):
            permutation = permutations[j, :]
            labels_under_perm = torch.from_numpy(permutations[j, labels_single.data.cpu().numpy().astype(int)])
            loss_under_perm = criterion(pred_llh_single, labels_under_perm.type(dtype_l))

            if (j == 0):
                loss_single = loss_under_perm
            else:
                loss_single = torch.min(loss_single, loss_under_perm)

        loss += loss_single
    return loss

def compute_accuracy_multiclass(pred_llh, labels, n_classes):
    pred_llh = pred_llh.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    batch_size = pred_llh.shape[0]
    pred_labels = from_scores_to_labels_multiclass_batch(pred_llh)
    acc = 0
    permutations = permuteposs(n_classes)
    for i in range(batch_size):
        pred_labels_single = pred_labels[i, :]
        labels_single = labels[i, :]
        for j in range(permutations.shape[0]):
            permutation = permutations[j, :]
            labels_under_perm = permutations[j, labels_single.astype(int)]

            acc_under_perm = compute_accuracy_multiclass_batch(pred_labels_single, labels_under_perm)
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
                
