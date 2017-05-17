# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
This script generates a convolutional neural network based classifier to
detect X-ray astronomical cavities.

The codes are written under the structure designed by Theano and Lasagne.

Note
====
[2017-02-23] Add script to handle the imbalanced situation

References
==========
[1] Lasagne tutorial
    http://lasagne.readthedocs.io/en/latest/user/tutorial.html
[2] Theano tutorial
    http://www.deeplearning.net/software/theano/

Methods
=======
load_data: load the prepared dataset
cnn_build: build the cnn network
cnn_train: train the cnn network
cnn_test: test and estimate by the trained network
iterate_minibathces: a batch helper method
"""

import os
# import time
import pickle
import numpy as np
import scipy.io as sio

# import theano
# import theano.tensor as T
# import lasagne
from ConvNet import ConvNet


def load_data(inpath, ratio_train=0.8, ratio_val=0.2):
    """
    Load the prepared dataset, and reshape for granular cnn training

    Inputs
    ======
    inpath: str
        Path of the mat dataset
    ratio_train: float
        Ratio of training samples in the sample set, default as 0.8
    ratio_val: float
        Ratio of validation samples in the training set, default as 0.2

    Outputs
    =======
    x_train: np.ndarray
        The training data
    y_train: np.ndarray
        Labels for the training data
    x_val: np.ndarray
        The validation data
    y_val: np.ndarray
        Labels for the validation data
    x_test: np.ndarray
        The test data
    y_test: np.ndarray
        Labels for the test data
    boxsize: integer
        boxsize of the subimage
    """
    # load the dataset
    try:
        data = sio.loadmat(inpath)
    except IOError:
        print("Path does not exist.")
        return

    data_bkg = data['data_bkg']
    data_ext = data['data_ext']
    data_cav = data['data_cav']
    label_bkg = data['label_bkg']
    label_ext = data['label_ext']
    label_cav = data['label_cav']
    # boxsize
    box = data_bkg.shape[1]
    boxsize = int(np.sqrt(box))

    # separate the major set into subsets
    numgra = int(np.round(len(label_bkg) / len(label_cav)))

    # calc train, val, test amounts, and shuffle
    idx_bkg = np.random.permutation(len(label_bkg))
    idx_ext = np.random.permutation(len(label_ext))
    idx_cav = np.random.permutation(len(label_cav))

    numtrain_bkg = int(np.floor(len(label_bkg) * ratio_train))
    numtrain_ext = int(np.floor(len(label_ext) * ratio_train))
    numtrain_cav = int(np.floor(len(label_cav) * ratio_train))

    numval_bkg = int(np.floor(numtrain_bkg * ratio_val))
    numval_ext = int(np.floor(numtrain_ext * ratio_val))
    numval_cav = int(np.floor(numtrain_cav * ratio_val))

    # form dataset
    x_train_bkg = data_bkg[idx_bkg[0:numtrain_bkg], :]
    y_train_bkg = label_bkg[idx_bkg[0:numtrain_bkg], :]
    x_test_bkg = data_bkg[idx_bkg[numtrain_bkg:], :]
    y_test_bkg = label_bkg[idx_bkg[numtrain_bkg:], :]

    x_train_ext = data_ext[idx_ext[0:numtrain_ext], :]
    y_train_ext = label_ext[idx_ext[0:numtrain_ext], :]
    x_test_ext = data_ext[idx_ext[numtrain_ext:], :]
    y_test_ext = label_ext[idx_ext[numtrain_ext:], :]

    x_train_cav = data_cav[idx_cav[0:numtrain_cav], :]
    y_train_cav = label_cav[idx_cav[0:numtrain_cav], :]
    x_test_cav = data_cav[idx_cav[numtrain_cav:], :]
    y_test_cav = label_cav[idx_cav[numtrain_cav:], :]

    # val
    x_val = np.row_stack((x_train_bkg[0:numval_bkg, :],
                          x_train_ext[0:numval_ext, :],
                          x_train_cav[0:numval_cav, :]))
    y_val = np.row_stack((y_train_bkg[0:numval_bkg],
                          y_train_ext[0:numval_ext],
                          y_train_cav[0:numval_cav]))
    x_val_temp = x_val.reshape(-1, 1, boxsize, boxsize)
    x_val = x_val_temp.astype('float32')
    y_val = y_val[:, 0].astype('int32')

    # test
    x_test = np.row_stack((x_test_bkg, x_test_ext, x_test_cav))
    y_test = np.row_stack((y_test_bkg, y_test_ext, y_test_cav))
    x_test_temp = x_test.reshape(-1, 1, boxsize, boxsize)
    x_test = x_test_temp.astype('float32')
    y_test = y_test[:, 0].astype('int32')

    # train
    # To save the memory, only output the idx
    cav_temp = x_train_cav[numval_cav:, :]
    cav_re = cav_temp.reshape(-1, 1, boxsize, boxsize)
    x_cav = cav_re.astype('float32')
    y_cav = y_train_cav[numval_cav:, ].astype('int32')

    ext_temp = x_train_ext[numval_ext:, :]
    ext_re = ext_temp.reshape(-1, 1, boxsize, boxsize)
    x_ext = ext_re.astype('float32')
    y_ext = y_train_ext[numval_ext:, ].astype('int32')

    bkg_temp = x_train_bkg[numval_bkg:, :]
    bkg_re = bkg_temp.reshape(-1, 1, boxsize, boxsize)
    x_bkg = bkg_re.astype('float32')
    y_bkg = y_train_bkg[numval_bkg:, ].astype('int32')

    x_train = {'bkg': x_bkg, 'cav': x_cav, 'ext': x_ext, 'numgra': numgra}
    y_train = {'bkg': y_bkg, 'cav': y_cav, 'ext': y_ext, 'numgra': numgra}

    return x_train, y_train, x_val, y_val, x_test, y_test, boxsize


def get_vote(inpath, numgra, data):
    """
    Estimate label using the granularied networks

    Inputs
    ======
    inpath: str
        Path to save the trained models
    numgra: integer
        Number of granule
    data: np.ndarray
        The data to test

    Ouput
    =====
    label_est: np.ndarray
        The estimated voted label
    """
    # Init
    cnn = ConvNet()
    numsample = data.shape[0]
    label = np.zeros((numsample, numgra))

    for i in range(numgra):
        print("Estimating by granula %d" % (i + 1))
        modelname = ("model%d.pkl" % (i + 1))
        modelpath = os.path.join(inpath, modelname)
        # load net
        cnn.net = cnn_load(modelpath)
        # est
        label[:, i] = cnn.cnn_predict(data)

    # vote
    idx_ext = np.where(label == 2)
    label[idx_ext] = 0

    label_sum = np.sum(label, axis=1)
    label_est = np.zeros((numsample,))
    thrs = numgra // 2
    label_est[np.where(label_sum >= thrs)] = 1

    return label_est


def cnn_load(modelpath):
    """
    Load the saved model

    Input
    =====
    modelpath: str
        Path to load the saved model

    Output
    ======
    model: dict
        The dictonary that saved the network and functions.
    """
    # Init
    fp = open(modelpath, 'rb')

    # load
    model = pickle.load(fp)

    # close
    fp.close()

    return model


def get_assess(img_mask, img_re):
    """
    Calculate performance of the approach

    Inputs
    ======
    img_mask: np.ndarray
        The mask image
    img_re: np.ndarray
        The recovered image

    Outputs
    =======
    r_sen: float
        Sensitivity, TP / (TP + FN)
    r_spe: float
        Specificity, TN / (TN + FP)
    r_acc: float
        Accuracy, (TP+TN)/All
    """
    Num_pos = len(np.where(img_re == 1)[0])
    Num_neg = len(np.where(img_re == 0)[0])

    # TP
    img_mul = img_mask * img_re
    TP = len(np.where(img_mul == 1)[0])
    # TN
    img_mul_rev = (img_mask - 1) * (img_re - 1)
    TN = len(np.where(img_mul_rev == 1)[0])

    # Acc
    r_acc = (TP + TN) / (Num_pos + Num_neg)
    # Sen
    r_sen = TP / Num_pos
    # Spe
    r_spe = TN / Num_neg

    return r_acc, r_sen, r_spe
