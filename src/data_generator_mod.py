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
from load import get_Pm, get_Pd, get_W_lg


class Generator(object):
    def __init__(self, N_train=50, N_test=100, generative_model='SBM_multiclass', p_SBM=0.8, q_SBM=0.2, n_classes=2, path_dataset='', num_examples_train=100, num_examples_test=10):
        self.N_train = N_train
        self.N_test = N_test
        # self.generative_model = 'ErdosRenyi'
        self.generative_model = generative_model
        # self.edge_density = 0.2
        # self.random_noise = False
        # self.noise = 0.03
        # self.noise_model = 2
        self.p_SBM = p_SBM
        self.q_SBM = q_SBM
        self.n_classes = n_classes
        self.path_dataset = path_dataset
        self.data_train = None
        self.data_test = None
        self.num_examples_train = num_examples_train
        self.num_examples_test = num_examples_test

    def SBM(self, p, q, N):
        W = np.zeros((N, N))

        p_prime = 1 - np.sqrt(1 - p)
        q_prime = 1 - np.sqrt(1 - q)

        n = N // 2

        W[:n, :n] = np.random.binomial(1, p, (n, n))
        W[n:, n:] = np.random.binomial(1, p, (N-n, N-n))
        W[:n, n:] = np.random.binomial(1, q, (n, N-n))
        W[n:, :n] = np.random.binomial(1, q, (N-n, n))
        W = W * (np.ones(N) - np.eye(N))
        W = np.maximum(W, W.transpose())

        perm = torch.randperm(N).numpy()
        blockA = perm < n
        labels = blockA * 2 - 1

        W_permed = W[perm]
        W_permed = W_permed[:, perm]
        return W_permed, labels

    def SBM_multiclass(self, p, q, N, n_classes):
        p_prime = 1 - np.sqrt(1 - p)
        q_prime = 1 - np.sqrt(1 - q)

        prob_mat = np.ones((N, N)) * q_prime

        n = N // n_classes

        for i in range(n_classes):
            prob_mat[i * n : (i+1) * n, i * n : (i+1) * n] = p_prime

        # print ('prob mat', prob_mat)

        W = np.random.rand(N, N) < prob_mat
        W = W.astype(int)

        W = W * (np.ones(N) - np.eye(N))
        W = np.maximum(W, W.transpose())

        perm = torch.randperm(N).numpy()
        # blockA = perm < n
        # labels = blockA * 2 - 1
        labels = (perm // n)

        W_permed = W[perm]
        W_permed = W_permed[:, perm]
        return W_permed, labels

    def ErdosRenyi(self, p, N):
        W = np.zeros((N, N))
        for i in range(0, N - 1):
            for j in range(i + 1, N):
                add_edge = (np.random.uniform(0, 1) < p)
                if add_edge:
                    W[i, j] = 1
                W[j, i] = W[i, j]
        return W

    def ErdosRenyi_netx(self, p, N):
        g = networkx.erdos_renyi_graph(N, p)
        W = networkx.adjacency_matrix(g).todense().astype(float)
        W = np.array(W)
        return W

    def RegularGraph_netx(self, p, N):
        """ Generate random regular graph """
        d = p * N
        d = int(d)
        g = networkx.random_regular_graph(d, N)
        W = networkx.adjacency_matrix(g).todense().astype(float)
        W = np.array(W)
        return W

    def create_dataset(self, directory, is_training):
        if (self.generative_model == 'SBM_multiclass'):
            if not os.path.exists(directory):
                os.mkdir(directory)
            if is_training:
                graph_size = self.N_train
                num_graphs = self.num_examples_train
            else:
                graph_size = self.N_test
                num_graphs = self.num_examples_test
            dataset = []
            for i in range(num_graphs):
                W, labels = self.SBM_multiclass(self.p_SBM, self.q_SBM, graph_size, self.n_classes)
                Pm = get_Pm(W)
                Pd = get_Pd(W)
                NB = get_W_lg(W)
                example = {}
                example['W'] = W
                example['labels'] = labels
                # example['Pm'] = Pm
                # example['Pd'] = Pd
                # example['NB'] = NB
                dataset.append(example)
            if is_training:
                print ('Saving the training dataset')
            else:
                print ('Saving the testing dataset')
            np.save(directory + '/dataset.npy', dataset)
            if is_training:
                self.data_train = dataset
            else:
                self.data_test = dataset
        else:
            raise ValueError('Generative model {} not supported'.format(self.generative_model))



    def prepare_data(self):
        train_directory = self.generative_model + '_nc' + str(self.n_classes) + '_p' + str(self.p_SBM) + '_q' + str(self.q_SBM) + '_gstr' + str(self.N_train) + '_numtr' + str(self.num_examples_train)
        
        train_path = os.path.join(self.path_dataset, train_directory)
        if os.path.exists(train_path + '/dataset.npy'):
            print('Reading training dataset at {}'.format(train_path))
            # self.data_train = self.load_from_directory(train_path)
            self.data_train = np.load(train_path + '/dataset.npy')
        else:
            print('Creating training dataset.')
            self.create_dataset(train_path, is_training=True)
            # print('Saving training datatset at {}'.format(train_path))
            # np.save(path, self.data_train)
        # load test dataset
        test_directory = self.generative_model + '_nc' + str(self.n_classes) + '_p' + str(self.p_SBM) + '_q' + str(self.q_SBM) + '_gste' + str(self.N_test) + '_numte' + str(self.num_examples_test)
        test_path = os.path.join(self.path_dataset, test_directory)
        if os.path.exists(test_path + '/dataset.npy'):
            print('Reading testing dataset at {}'.format(test_path))
            # self.data_test = self.load_from_directory(test_path)
            self.data_test = np.load(test_path + '/dataset.npy')
        else:
            print('Creating testing dataset.')
            self.create_dataset(test_path, is_training=False)
            # print('Saving testing datatset at {}'.format(test_path))
            # np.save(open(path, 'wb'), self.data_test)

    def sample_single(self, i, is_training=True):
        if is_training:
            dataset = self.data_train
        else:
            dataset = self.data_test
        example = dataset[i]
        if (self.generative_model == 'SBM_multiclass'):
            W_np = example['W']
            labels = np.expand_dims(example['labels'], 0)
            labels_var = Variable(torch.from_numpy(labels), volatile=not is_training)
            return W_np, labels_var 

    def sample_otf_single(self, is_training=True, cuda=True):
        if is_training:
            N = self.N_train
        else:
            N = self.N_test
        if self.generative_model == 'SBM':
            W, labels = self.SBM(self.p_SBM, self.q_SBM, N)
        elif self.generative_model == 'SBM_multiclass':
            W, labels = self.SBM_multiclass(self.p_SBM, self.q_SBM, N, self.n_classes)
        else:
            raise ValueError('Generative model {} not supported'.format(self.generative_model))
        # if self.random_noise:
        #     self.noise = np.random.uniform(0.000, 0.050, 1)
        # if self.noise_model == 1:
        #     # use noise model from [arxiv 1602.04181], eq (3.8)
        #     noise = self.ErdosRenyi(self.noise, self.N_train)
        #     W_noise = W*(1-noise) + (1-W)*noise
        # elif self.noise_model == 2:
        #     # use noise model from [arxiv 1602.04181], eq (3.9)
        #     pe1 = self.noise
        #     pe2 = (self.edge_density*self.noise)/(1.0-self.edge_density)
        #     noise1 = self.ErdosRenyi_netx(pe1, self.N_train)
        #     noise2 = self.ErdosRenyi_netx(pe2, self.N_train)
        #     W_noise = W*(1-noise1) + (1-W)*noise2
        # else:
        #     raise ValueError('Noise model {} not implemented'.format(self.noise_model))
        labels = np.expand_dims(labels, 0)
        labels = Variable(torch.from_numpy(labels), volatile=not is_training)
        W = np.expand_dims(W, 0)
        return W, labels



if __name__ == '__main__':
    ###################### Test Generator module ##############################
    path = '/home/chenzh/tmp/'
    gen = Generator(path)
    gen.num_examples_train = 10
    gen.num_examples_test = 10
    gen.N = 50
    # gen.generative_model = 'Regular'
    gen.generative_model = 'SBM'
    gen.load_dataset()
    g1, g2 = gen.sample_batch(32, cuda=False)
    print(g1[0].size())
    print(g1[1][0].data.cpu().numpy())
    W = g1[0][0, :, :, 1]
    W_noise = g2[0][0, :, :, 1]
    print(W, W.size())
    print(W_noise.size(), W_noise)
    ################### Test graph generators networkx ########################
    # path = '/home/anowak/tmp/'
    # gen = Generator(path)
    # p = 0.2
    # N = 50
    # # W = gen.ErdosRenyi_netx(p, N)
    # W = gen.RegularGraph_netx(3, N)
    # G = networkx.from_numpy_matrix(W)
    # networkx.draw(G)
    # # plt.draw(G)
    # plt.savefig('/home/anowak/tmp/prova.png')
    # print('W', W)
