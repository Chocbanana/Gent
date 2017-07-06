#! python3.6

"""
Author: Bhavana Jonnalagadda, 2017
"""

import math
from typing import *
import numpy as np

from EvolutionaryAlg import *


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


"""
TODO Later (NOT NOW!!) :
- Type definitions
- Documentation (properly formatted)
"""



# TODO: IMP redo for new "same" gene structure ie no recursive updating for output
# TODO: add checks/asserts for correct params in genes
class NetworkNode:
    def __init__(self, gene):
        self.gene = gene
        # Input Connections
        self.input = []
        # Output connections
        self.output = []
        self.input_tensor = None
        self.tensor = None

    def add_input(self, input_list):
        for node in input_list:
            if node not in self.input:
                self.input.append(node)
                node.add_output([self])
                self._build_tensor()

    def remove_input(self, input_list):
        for node in input_list:
            if node in self.input:
                self.input.remove(node)
                node.remove_output([self])
                self._build_tensor()

    def add_output(self, output_list):
        for node in output_list:
            if node not in self.output:
                self.output.append(node)
                node.add_input([self])

    def remove_output(self, output_list):
        for node in output_list:
            if node in self.output:
                self.output.remove(node)
                node.remove_input([self])

    def remove_all(self):
        self.remove_input(self.input)
        self.remove_output(self.output)


class NetworkInput(NetworkNode):
    def __init__(self, gene, output_list=()):
        super(NetworkInput, self).__init__(gene)

        self._build_tensor()

        if output_list: self.add_output(output_list)

    def _build_tensor(self):
        self.tensor = tf.placeholder("float", name="input")


class NetworkOutput(NetworkNode):
    gene_space = {"nodetype": ["noutput"],
                  "input": ["conv", "recurrent", "maxpool", "matmul"],
                  "output": []}

    """
    Required params: nodetype, activation
    """

    def __init__(self, gene, input_list=[]):
        super(NetworkOutput, self).__init__(gene)

        # self.biasvar = tf.Variable(tf.constant([1.0]))
        # self.biasvar2 = tf.Variable(tf.constant([1.0]))

        if input_list: self.add_input(input_list)

    def _build_tensor(self):
        # NOTE: Assumes dim n x m x k x ..., giving  n x m x sum(k) x ...
        if len(self.input) == 1:
            self.input_tensor = self.input[0].tensor
        else:
            self.input_tensor = tf.concat(axis=2, values=[n.tensor for n in self.input])
        del self.tensor
        # NOTE: Assumes dim n x m x k x 1, reducing to n x m
        # self.tensor = self.gene["activation"](tf.reduce_sum(self.input_tensor, [2, 3]))
        if "activation" in self.gene and self.gene["activation"] is not None:
            self.tensor = self.gene["activation"](self.input_tensor)

        else:
            self.tensor = self.input_tensor


class ConvLayer(NetworkNode):
    gene_space = {"nodetype": ["conv"],
                  "input": ["conv", "recurrent", "ninput", "maxpool", "matmul"],
                  "output": ["conv", "recurrent", "noutput", "maxpool", "matmul"]}
    """
    Required params: nodetype, filtersize
    Optional: activation
    """

    def __init__(self, gene, input_list=[], output_list=[]):
        super(ConvLayer, self).__init__(gene)

        # if "channels" in self.gene and self.gene["channels"] is not None:
        #     ch = self.gene["channels"]
        #     self.weights = tf.Variable(tf.random_uniform([self.gene["filtersize"], 2, ch[0], 1], minval=-1.0, maxval=1.0))
        #     self.weights2 = tf.Variable(tf.random_uniform([1, 1, *ch], minval=-1.0, maxval=1.0))
        # else:
        self.weights = tf.Variable(tf.random_uniform(self.gene["filtersize"], minval=-1.0, maxval=1.0))
        self.biasvar = tf.Variable(tf.constant([0.1], shape=[self.gene["filtersize"][3]]))

        if input_list: self.add_input(input_list)
        if output_list: self.add_output(output_list)

    def _build_tensor(self):
        # NOTE: Assumes dim n x m x k x ..., giving  n x m x sum(k) x ...
        if len(self.input) == 1:
            self.input_tensor = self.input[0].tensor
        else:
            self.input_tensor = tf.concat(axis=2, values=[n.tensor for n in self.input])

        # NOTE: Assumes dim n x m x k x 1, reducing to n x m
        del self.tensor
        if "activation" in self.gene and self.gene["activation"] is not None:
            self.tensor = self.gene["activation"](self._conv(self.input_tensor, self.weights, self.biasvar))
        else:
            self.tensor = self._conv(self.input_tensor, self.weights, self.biasvar)

        for node in self.output: node.add_input([self])

    def _conv(self, a, b, c):
        return tf.nn.conv2d(a, b, strides=[1, 1, 1, 1], padding='SAME') + c


class MaxPool(NetworkNode):
    gene_space = {"nodetype": ["maxpool"],
                  "input": ["conv", "recurrent", "ninput", "maxpool", "matmul"],
                  "output": ["conv", "recurrent", "noutput", "maxpool", "matmul"]}
    """
    Required params: nodetype
    Optional: activation
    """

    def __init__(self, gene, input_list=[], output_list=[]):
        super(MaxPool, self).__init__(gene)

        if input_list: self.add_input(input_list)
        if output_list: self.add_output(output_list)

    def _build_tensor(self):
        # NOTE: Assumes dim n x m x k x ..., giving  n x m x sum(k) x ...
        if len(self.input) == 1:
            self.input_tensor = self.input[0].tensor
        else:
            self.input_tensor = tf.concat(axis=2, values=[n.tensor for n in self.input])

        # NOTE: Assumes dim n x m x k x 1, reducing to n x m
        del self.tensor
        if "activation" in self.gene and self.gene["activation"] is not None:
            self.tensor = self.gene["activation"](self._pool(self.input_tensor))
        else:
            self.tensor = self._pool(self.input_tensor)

        for node in self.output: node.add_input([self])

    def _pool(self, a):
        return tf.nn.max_pool(a, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name="maxpool")


class FullyConnected(NetworkNode):
    gene_space = {"nodetype": ["matmul"],
                  "input": ["conv", "recurrent", "ninput", "maxpool", "matmul"],
                  "output": ["conv", "recurrent", "noutput", "maxpool", "matmul"]}
    """
    Required params: nodetype, activation, n_dim, m_dim
    Optional:
    """

    def __init__(self, gene, input_list=[], output_list=[]):
        super(FullyConnected, self).__init__(gene)

        if input_list: self.add_input(input_list)
        if output_list: self.add_output(output_list)

    def _build_tensor(self):
        if not self.input:
            return None
        elif len(self.input) == 1:
            self.input_tensor = self.input[0].tensor
        else:
            self.input_tensor = tf.concat(axis=2, values=[n.tensor for n in self.input])

        # NOTE: n_dim MUST BE THE same as a1 x a2 x ... where [a0...an] are the
        # dimensions of the input
        self.n = self.gene["n_dim"]
        self.m = self.gene["m_dim"]

        self.weights = tf.Variable(tf.random_normal([self.n * len(self.input), self.m]))
        self.biasvar = tf.Variable(tf.constant([0.1], shape=[self.m]))

        del self.tensor
        # NOTE: Reduces to -1 x m
        if "activation" in self.gene and self.gene["activation"] is not None:
            self.tensor = self.gene["activation"](self._matmul2d(self.input_tensor, self.weights, self.biasvar))
        else:
            self.tensor = self._matmul2d(self.input_tensor, self.weights, self.biasvar)

        for node in self.output: node.add_input([self])

    def _matmul2d(self, a, b, c):
        return tf.matmul(tf.reshape(a, [-1, self.n * len(self.input)]), b) + c


# TODO: Implement
# class ReccurrentLayer:
#     def __init__(self):
#         self.steps = 3


# TODO: redo, store nodes in dict w geneID as key
class NeuralNetwork(Phenotype):
    nodetypes = {"ninput": NetworkInput, "conv": ConvLayer,
                 "noutput": NetworkOutput, "maxpool": MaxPool,
                 "matmul": FullyConnected}

    # TODO: IMPORTANT fix the problem of empty inputs when removing them
    def __init__(self, genes_or_nodes):
        self.nodes = [(self.nodetypes[n["nodetype"]](n) if isinstance(n, dict) else n) for n in genes_or_nodes]
        for n in self.nodes:
            n.remove_all()
        for n in self.nodes:
            inlist = [x for x in self.nodes if x.gene["geneID"] in n.gene["input"]]
            outlist = [x for x in self.nodes if x.gene["geneID"] in n.gene["output"]]
            n.add_input(inlist)
            n.add_output(outlist)

    def compatibility(self, m1, m2):
        pass


class HyperParamOpt:
    """
    Hyperparameter trainer.

    :param params:
    :returns:
    """

    def __init__(self, params: NamedTuple):


# TODO: use
# regularizer = tf.contrib.layers.l1_regularizer(0.01, scope=None)
# tf.contrib.layers.apply_regularization(regularizer, weights_list=None)
# class NetworkRunner(ga.Evaluator):
# TODO: make much more flexible!!!
class NetworkRunner:
    def __init__(self, basedir, xtr, ytr, xte, yte):

        self.basedir = basedir
        self.x_train = xtr
        self.y_train = ytr
        self.x_test = xte
        self.y_test = yte

    def eval(self, network, gen, species):
        return self.train_network(network, "/gen" + str(gen) + "/sp" + str(species))

    def train_network(self, network, subdir='', epochs=50, batch_size=10, learn_rate=0.0001, **kwargs):

        x_ = next(n.tensor for n in network.nodes if n.gene["nodetype"] == "ninput")
        outnode, output = next((n, n.tensor) for n in network.nodes if n.gene["nodetype"] == "noutput")
        y_ = tf.placeholder("float", name="actual_spike_rate")

        # Error = MSE / Variance
        if "error" in kwargs:
            error = kwargs["error"](output, y_)
        else:
            error = tf.reduce_mean(tf.square(output - y_)) / tf.reduce_mean(tf.square(y_ - tf.reduce_mean(y_)))

        # experimental RNN layer
        # cell = tf.nn.rnn_cell.LSTMCell(1);
        # convs = [n.tensor for n in network.nodes if n.gene["nodetype"] == "conv"]
        # rnnout, state = tf.nn.dynamic_rnn(cell, convs[-1], dtype=tf.float64)
        # output2 = tf.squeeze(rnnout, axis=2)

        if "loss_opt" in kwargs:
            loss, optimizer = kwargs["loss_opt"](output, y_)
        else:
            # Loss = Poisson log likelihood
            loss = tf.nn.log_poisson_loss(tf.log(output), y_, compute_full_loss=False)
            # loss = tf.nn.l2_loss(output - y_)
            # Specifically for the reimplementation
            reg = tf.contrib.layers
            l2reg = reg.l2_regularizer(0.01)
            l1reg = reg.l1_regularizer(0.03)
            l2weights = [n.tensor for n in network.nodes if n.gene["nodetype"] == "conv"]
            l1weights = [n.tensor for n in network.nodes if n.gene["nodetype"] == "noutput"]
            loss = loss + l2reg(l2weights[0]) + l2reg(l2weights[1]) + l1reg(l1weights[0])

            # Optimizer = ADAM
            # optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)
            opt = tf.train.AdamOptimizer(learning_rate=learn_rate)
            # Compute the gradients for a list of variables.
            grads_and_vars = opt.compute_gradients(loss)

            # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
            # need to the 'gradient' part, for example cap them, etc.
            capped_grads_and_vars = [(tf.clip_by_average_norm(gv[0], 0.7), gv[1]) for gv in grads_and_vars]

            # Ask the optimizer to apply the capped gradients.
            optimizer = opt.apply_gradients(capped_grads_and_vars)

        # tensorboard_summaries = self._create_writers(output, loss, error)
        all_loss_error = np.empty([epochs, 3])
        # all_output = np.empty([epochs, self.x_test.shape[1]])


        custom_stats = [None] * epochs

        with tf.Session() as sess:
            # writer = tf.summary.FileWriter(self.basedir + subdir, sess.graph)
            sess.run([tf.global_variables_initializer()])

            for e in range(epochs):
                # create batches
                b = batch_size
                inds = np.random.permutation(range(len(self.x_train)))
                batches = [inds[i:i + b] for i in range(0, len(inds), b) if i + b < len(inds)]

                for i in range(len(batches)):
                    fd = {x_: self.x_train[batches[i], :, :, :], y_: self.y_train[batches[i], :]}
                    sess.run(optimizer, feed_dict=fd)
                    # debug = sess.run([error, tf.reduce_mean(loss), tf.reduce_mean(tf.nn.moments(output, [0])[0]), tf.reduce_mean(tf.nn.moments(output, [0])[1]), optimizer], feed_dict=fd)
                    # print("Batch %d Error: %f LOSS: %f, MEAN: %f, VAR: %f" % (i, debug[0], debug[1], debug[2], debug[3]))

                    # Write data
                    if (i % 500 == 0):
                        fd = {x_: self.x_test, y_: self.y_test}
                        stats = sess.run([tf.reduce_mean(loss), error, output], feed_dict=fd)
                        # writer.add_summary(stats[2])
                        print("epoch: %d  iter: %d loss: %f error: %f \n" % (e, i, stats[0], stats[1]))

                fd = {x_: self.x_test, y_: self.y_test}
                if "run_custom" in kwargs:
                    stats = sess.run([tf.reduce_mean(loss), error, output, *kwargs["run_custom"]], feed_dict=fd)
                    custom_stats[e] = stats[3:]
                else:
                    stats = sess.run([tf.reduce_mean(loss), error, output], feed_dict=fd)
                    # writer.add_summary(stats[3])
                    # writer.flush()

                all_loss_error[e, 0:2] = stats[0:2]
                corr = np.mean(np.corrcoef(stats[2], self.y_test))
                all_loss_error[e, 2] = corr
                # all_output[e, :] = stats[2]


                print("epoch: %d  loss: %f error: %f  corr: %f  \n" % (e, stats[0], stats[1], float(corr)))


                # writer.close()

        # if "run_custom" in kwargs:
        #     return all_loss_error, all_output, custom_stats
        # else:
        #     return all_loss_error, all_output

        if "run_custom" in kwargs:
            return all_loss_error, custom_stats
        else:
            return all_loss_error

    # def _create_stat_tensors(self, output,  learn_rate):
    #     y_ = tf.placeholder("float", name="actual_spike_rate")

    #     # Loss = Poisson log likelihood
    #     loss = tf.nn.log_poisson_loss(output, y_)
    #     # loss = tf.nn.l2_loss(output - y_)
    #     # Error = MSE / Variance
    #     error = tf.reduce_mean(tf.square(output - y_)) / tf.reduce_mean(tf.square(y_ - tf.reduce_mean(y_)))
    #     # Optimizer = ADAM
    #     optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)
    #     return y_, loss, error, optimizer

    def _create_writers(self, output, loss, error):

        output_summary = tf.summary.histogram('generated_spike_train', output)
        loss_summary = tf.summary.scalar('p_loss', tf.reduce_mean(loss))
        err_summary = tf.summary.scalar('mse_error', error)
        return tf.summary.merge([output_summary, loss_summary, err_summary])
