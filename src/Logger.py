import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from losses import compute_loss_multiclass, compute_accuracy_multiclass

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    # torch.manual_seed(0)

def compute_recovery_rate(pred, labels):
    pred = pred.max(2)[1]
    error = 1 - torch.eq(pred, labels).type(dtype)#.squeeze(2)
    frob_norm = error.mean(1)#.squeeze(1)
    accuracy = 1 - frob_norm
    accuracy = accuracy.mean(0).squeeze()
    return accuracy.data.cpu().numpy()#[0]

def from_scores_to_labels(pred):
    pred = pred.squeeze(0)
    labels_pred = (pred[:, 0] > pred[:, 1]).type(dtype) * 2.0 - 1
    return labels_pred

def compute_accuracy_bcd(labels_pred, labels):
    labels = labels.squeeze(0)
    # print (pred)
    # print (labels_pred)
    # print (labels)
    dot_product = labels_pred.dot(labels.type(dtype)).data
    if torch.cuda.is_available():
        dot_product = dot_product.cpu()
    acc = abs(dot_product.numpy() / labels.data.shape[0])
    return acc

def from_scores_to_labels_batch(pred):
    # pred = pred.squeeze(0)
    labels_pred = (pred[:, :, 0] > pred[:, :, 1]).type(dtype) * 2.0 - 1
    return labels_pred.unsqueeze(1)

def from_scores_to_labels_mcd_batch(pred):
    # pred = pred.squeeze(0)
    # labels_pred = (pred[:, :, 0] > pred[:, :, 1]).type(dtype) * 2.0 - 1
    labels_pred = np.argmax(pred, axis = 2).astype(int)
    return labels_pred

def from_scores_to_labels_batch_2(pred):
    # pred = pred.squeeze(0)
    pred_sorted, indices = pred[:, :, 0].sort(1)
    labels_pred = torch.zeros(pred.data.shape[0], 1, pred.data.shape[1]).type(dtype)
    for i in range(pred.data.shape[0]):
        mid_pt = pred.data.shape[1] // 2
        if torch.cuda.is_available():
            indices = indices.cpu()
        indices_for_plus1 = indices.data.numpy()[i, :pred.data.shape[1] // 2].tolist()
        indices_for_neg1 = indices.data.numpy()[i, pred.data.shape[1] // 2 :].tolist()
        labels_pred_ith = labels_pred[i, :, :]
        labels_pred_ith[:, indices_for_plus1] = 1
        labels_pred_ith[:, indices_for_neg1] = -1
        labels_pred[i, :, :] = labels_pred_ith
        #
        # labels_pred[i, :, indices.data.numpy()[i, :pred.shape[1] // 2].tolist()] = 1
        # labels_pred[i, :, indices.data.numpy()[i, pred.shape[1] // 2:].tolist()] = -1
    # labels_pred = (pred[:, :, 0] > pred[:, :, 1]).type(dtype) * 2.0 - 1
    return Variable(labels_pred)

def compute_accuracy_bcd_batch(labels_pred, labels):
    # labels = labels.squeeze(0)
    # print (pred)
    # print (labels_pred)
    # print (labels)
    # dot_product = labels_pred.dot(labels.type(dtype)).data

    # print (labels.data.shape)
    labels = labels.unsqueeze(2)
    # print (labels_pred.data.shape)
    # print (labels.data.shape)
    # labels.unsqueeze(2)
    # print (labels.data.shape)
    # print ('labels_pred', labels_pred)
    # print ('labels', labels)
    dot_product = torch.bmm(labels_pred, labels.type(dtype)).data
    # print ('dot product', dot_product)
    # for i in range(labels.data.shape[0]):
    #     print ('labels_pred', labels_pred[i].squeeze(0))
    #     print ('labels_true', labels[i].squeeze(1))
    if torch.cuda.is_available():
        dot_product = dot_product.cpu()
    acc = np.mean(abs(dot_product.numpy()) / labels.data.shape[1])
    return acc

def compute_accuracy_mcd_batch(labels_pred, labels):
    overlap = (labels_pred == labels).astype(int)
    acc = np.mean(labels_pred == labels)
    return acc

class Logger(object):
    def __init__(self, path_logger):
        directory = os.path.join(path_logger, 'plots/')
        self.path = path_logger
        self.path_dir = directory
        # Create directory if necessary
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        self.loss_train = []
        self.loss_test = []
        self.accuracy_train = []
        self.accuracy_test = []
        self.args = None

    def write_settings(self, args):
        self.args = {}
        # write info
        path = os.path.join(self.path, 'experiment.txt')
        with open(path, 'w') as file:
            for arg in vars(args):
                file.write(str(arg) + ' : ' + str(getattr(args, arg)) + '\n')
                self.args[str(arg)] = getattr(args, arg)

    def save_model(self, model):
        save_dir = os.path.join(self.path, 'parameters/')
        # Create directory if necessary
        try:
            os.stat(save_dir)
        except:
            os.mkdir(save_dir)
        path = os.path.join(save_dir, 'gnn.pt')
        torch.save(model, path)
        print('Model Saved.')

    def load_model(self):
        load_dir = os.path.join(self.path, 'parameters/')
        # check if any training has been done before.
        try:
            os.stat(load_dir)
        except:
            print("Training has not been done before testing. This session will be terminated.")
            sys.exit()
        path = os.path.join(load_dir, 'gnn.pt')
        print('Loading the most recent model...')
        siamese_gnn = torch.load(path)
        return siamese_gnn

    def add_train_loss(self, loss):
        self.loss_train.append(loss.data.cpu().numpy())

    def add_test_loss(self, loss):
        self.loss_test.append(loss.data.cpu().numpy())

    def add_train_accuracy(self, pred, labels):
        accuracy = compute_recovery_rate(pred, labels)
        self.accuracy_train.append(accuracy)

    def add_train_accuracy_bcd(self, pred, labels):
        labels_pred = from_scores_to_labels_batch(pred)
        accuracy = compute_accuracy_bcd_batch(labels_pred, labels)
        self.accuracy_train.append(accuracy)

    def add_train_accuracy_mcd(self, pred, labels, n_classes):
        # pred = pred.data.numpy()
        # labels = labels.data.numpy()
        # labels_pred = from_scores_to_labels_mcd_batch(pred)
        # accuracy = compute_accuracy_mcd_batch(labels_pred, labels)
        # self.accuracy_train.append(accuracy)

        accuracy = compute_accuracy_multiclass(pred, labels, n_classes)
        self.accuracy_train.append(accuracy)

    def add_test_accuracy(self, pred, labels):
        accuracy = compute_recovery_rate(pred, labels)
        self.accuracy_test.append(accuracy)

    def add_test_accuracy_bcd(self, pred, labels):
        # labels_pred = from_scores_to_labels_batch_2(pred)
        labels_pred = from_scores_to_labels_batch(pred)
        accuracy = compute_accuracy_bcd_batch(labels_pred, labels)
        self.accuracy_test.append(accuracy)

    def add_test_accuracy_mcd(self, pred, labels, n_classes):
        accuracy = compute_accuracy_multiclass(pred, labels, n_classes)
        self.accuracy_test.append(accuracy)


    def plot_train_loss(self):
        plt.figure(0)
        plt.clf()
        iters = range(len(self.loss_train))
        plt.semilogy(iters, self.loss_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Loss: p={}, p_e={}'
                  .format(self.args['edge_density'], self.args['noise']))
        path = os.path.join(self.path_dir, 'training_loss.png')
        plt.savefig(path)

    def plot_test_loss(self):
        plt.figure(1)
        plt.clf()
        test_freq = self.args['test_freq']
        iters = test_freq * np.arange(len(self.loss_test))
        plt.semilogy(iters.tolist(), self.loss_test, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Testing Loss: p={}, p_e={}'
                  .format(self.args['edge_density'], self.args['noise']))
        path = os.path.join(self.path_dir, 'testing_loss.png')
        plt.savefig(path)

    def plot_train_accuracy(self):
        plt.figure(0)
        plt.clf()
        iters = range(len(self.accuracy_train))
        plt.plot(iters, self.accuracy_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy: p={}, p_e={}'
                  .format(self.args['edge_density'], self.args['noise']))
        path = os.path.join(self.path_dir, 'training_accuracy.png')
        plt.savefig(path)

    def plot_test_accuracy(self):
        plt.figure(1)
        plt.clf()
        test_freq = self.args['test_freq']
        iters = test_freq * np.arange(len(self.accuracy_test))
        plt.plot(iters.tolist(), self.accuracy_test, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Testing Accuracy: p={}, p_e={}'
                  .format(self.args['edge_density'], self.args['noise']))
        path = os.path.join(self.path_dir, 'testing_accuracy.png')
        plt.savefig(path)

    def save_results(self):
        path = os.path.join(self.path, 'results.npz')
        np.savez(path, accuracy_train=np.array(self.accuracy_train),
                 accuracy_test=np.array(self.accuracy_test),
                 loss_train=self.loss_train, loss_test=self.loss_test)
