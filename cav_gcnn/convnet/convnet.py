# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
A convolutional neural network (CNN) constructor based on theano, lasagne, and
nolearn

Reference
=========
[1] convolutional_autoencoder (mikesj)
    https://github.com/mikesj-public/convolutional_autoencoder
[2] nolearn.lasagne
    http://pythonhosted.org/nolearn/lasagne.html
"""

import lasagne
# from lasagne.layers import get_output
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from nolearn.lasagne import NeuralNet, BatchIterator


class ConvNet():
    """
    A convolutional neural network (CNN) constructor

    inputs
    ======
    X_in: np.ndarray
        The sample matrix, whose size is (s,d,r,c).
        s: number of samples, d: dimensions
        r: rows, c: cols
    X_out: np.ndarray
        The corresponding label (category) of the X_in.
    kernel_size: list
        Box sizes of the kernels in each ConvLayer
    kernel_num: list
        Number of kernels in each ConvLayer
    pool_flag: list of bool values
        Flags of pooling layer w.r.t. to the ConvLayer
    fc_nodes: list
        The dense layers after the full connected layer
        of last ConvLayer or pooling layer.

    methods
    =======
    gen_layers: construct the layers
    cnn_build: build the cnn network
    cnn_train: train the cnn network
    cnn_eval: evaluate the cnn network
    cnn_save: save the network
    """

    def __init__(self, X_in=None, X_out=None, numclass=3, kernel_size=[2, 3, 4],
                 kernel_num=[15, 15, 15], pool_flag=[False, False, False],
                 fc_nodes=None):
        """
        The initializer
        """
        self.X_in = X_in
        self.X_out = X_out
        self.numclass = numclass
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.pool_flag = pool_flag
        self.pool_size = 2
        self.fc_nodes = fc_nodes

    def gen_BatchIterator(self, batch_size=100, shuffle=True):
        """Generate the batch iterator"""
        B = BatchIterator(batch_size=batch_size, shuffle=shuffle)
        return B

    def gen_layers(self, droprate=0.5):
        """Construct the layers"""

        # Init <TODO>
        pad_in = 'valid'
        self.layers = []
        # input layer
        l_input = (InputLayer,
                   {'shape': (None, self.X_in.shape[1],
                              self.X_in.shape[2],
                              self.X_in.shape[3])})
        self.layers.append(l_input)
        # Conv and pool layers
        rows, cols = self.X_in.shape[2:]
        for i in range(len(self.kernel_size)):
            # conv
            l_conv = (Conv2DLayer,
                      {'num_filters': self.kernel_num[i],
                          'filter_size': self.kernel_size[i],
                          'nonlinearity': lasagne.nonlinearities.rectify,
                          'W': lasagne.init.GlorotUniform(),
                          'pad': pad_in})
            self.layers.append(l_conv)
            rows = rows - self.kernel_size[i] + 1
            cols = cols - self.kernel_size[i] + 1
            # pool
            if self.pool_flag[i]:
                l_pool = (MaxPool2DLayer,
                          {'pool_size': self.pool_size})
                self.layers.append(l_pool)
                rows = rows // 2
                cols = cols // 2
        # dropout
        l_drop = (DropoutLayer, {'p': droprate})
        # self.layers.append(l_drop)
        # full connected layer
        num_fc = rows * cols * self.kernel_num[-1]
        l_fc = (DenseLayer,
                {'num_units': num_fc,
                 'nonlinearity': lasagne.nonlinearities.rectify,
                 'W': lasagne.init.GlorotUniform(),
                 'b': lasagne.init.Constant(0.)}
                )
        self.layers.append(l_fc)
        # dense
        if not self.fc_nodes is None:
            for i in range(len(self.fc_nodes)):
                self.layers.append(l_drop)
                l_dense = (DenseLayer, {'num_units': self.fc_nodes[i]})
                self.layers.append(l_dense)
        # output layer
        self.layers.append(l_drop)
        l_out = (DenseLayer,
                 {'name': 'output',
                  'num_units': self.numclass,
                  'nonlinearity': lasagne.nonlinearities.softmax})
        self.layers.append(l_out)

    def cnn_build(self, max_epochs=20, batch_size=100,
                  learning_rate=0.001, momentum=0.9,
                  verbose=1):
        """Build the network"""
        if batch_size is None:
            self.net = NeuralNet(
                layers=self.layers,
                max_epochs=max_epochs,
                update=lasagne.updates.nesterov_momentum,
                update_learning_rate=learning_rate,
                update_momentum=momentum,
                regression=False,
                verbose=verbose)
        else:
            # batch iterator
            batch_iterator = self.gen_BatchIterator(batch_size=batch_size)
            self.net = NeuralNet(
                layers=self.layers,
                batch_iterator_train=batch_iterator,
                max_epochs=max_epochs,
                update=lasagne.updates.nesterov_momentum,
                update_learning_rate=learning_rate,
                update_momentum=momentum,
                regression=False,
                verbose=verbose)

    def cnn_train(self):
        """Train the cae net"""
        print("Training the network...")
        self.net.fit(self.X_in, self.X_out)
        print("Training done.")

    def cnn_eval(self):
        """Draw evaluation lines
        <TODO>
        """
        from nolearn.lasagne.visualize import plot_loss
        plot_loss(self.net)

    def cnn_predict(self, img):
        """
        Predict the output of the input image

        input
        =====
        img: np.ndarray
            The image matrix, (r,c)

        output
        ======
        label_pred: np.ndarray
            The predicted image matrix
        """
        if len(img.shape) == 4:
            rows = img.shape[2]
            cols = img.shape[3]
        elif len(img.shape) == 3:
            rows = img.shape[1]
            cols = img.shape[2]
            img = img.reshape(img.shape[0], 1, rows, cols)
        elif len(img.shape) == 2:
            rows, cols = img.shape
            img = img.reshape(1, 1, rows, cols)
        else:
            print("The shape of image should be 2 or 3 d")
        label_pred = self.net.predict(img)

        return label_pred

    def cnn_save(self, savepath='cnn.pkl'):
        """Save the trained network

        input
        =====
        savepath: str
            Path of the net to be saved
        """
        import sys
        sys.setrecursionlimit(1000000)
        import pickle
        fp = open(savepath, 'wb')
        # write
        pickle.dump(self.net, fp)
        fp.close()
