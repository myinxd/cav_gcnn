# !/usr/bin/env python3
# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

"""
Load data, and train the granular CNN network.

Note
====
Model and parameters are saved as 'model*.pkl' and 'params*.mat'
"""

import os
import argparse

import numpy as np

from cav_gcnn.utils import utils_cnn as utils
from cav_gcnn.convnet import ConvNet

def vec2mat(vec, numclass):
    """Trainform the vector like label to (-1, numclass) matrix"""
    vec = vec[:,0].astype(int)
    mat = np.zeros((vec.shape[0],numclass))
    for i in range(vec.shape[0]):
        mat[i, vec[i]] = 1

    return mat.astype('float32')

def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(
        description="Load data, and train the network.")
    # Parameters
    parser.add_argument("inpath", help="path of the samples")
    parser.add_argument("outpath", help="path to save result.")
    parser.add_argument("numepoch", help="Number of epochs")
    args = parser.parse_args()

    inpath = args.inpath
    outpath = args.outpath
    numepoch = int(args.numepoch)

    if not os.path.exists(inpath):
        print("The inpath does not exist.")
        return

    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test, boxsize = utils.load_data(
        inpath=inpath, ratio_train=0.9, ratio_val=0.2)
    # build the network
    # train
    # numgra = x_train['numgra']
    numgra = 5
    cav_data = x_train['cav']
    ext_data = x_train['ext']
    bkg_data = x_train['bkg']
    cav_label = y_train['cav']
    ext_label = y_train['ext']
    bkg_label = y_train['bkg']


    for i in range(5):
        print("Training granular %d of %d..." % (i + 1, numgra))
        data_train = np.row_stack((bkg_data[0 + i::numgra, ],
                                   cav_data, cav_data, ext_data))
        label_train = np.row_stack((bkg_label[0 + i::numgra, ],
                                    cav_label, cav_label, ext_label))
        label_train = label_train[:,0].astype('int32')
        print(data_train.shape)
        print(label_train.shape)
        # Contruct the network
        net = ConvNet(X_in=data_train, X_out=label_train, fc_nodes=[128])
        net.gen_layers()
        # build
        print("Building the network...")
        net.cnn_build(max_epochs=numepoch,learning_rate=0.001,momentum=0.95)
        print("Building done.")
        # train
        net.cnn_train()
        # save model
        modelname = ("model%d.pkl" % (i + 1))
        savepath = os.path.join(outpath, modelname)
        net.cnn_save(savepath=savepath)

if __name__ == "__main__":
    main()
