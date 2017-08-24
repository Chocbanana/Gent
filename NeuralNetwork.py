#! python3.6

"""
Author: Bhavana Jonnalagadda, 2017
"""
import os
import math
import pickle
import random
from typing import *
import numpy as np

# from EvolutionaryAlg import *
import NNGenome as gn


import torch, torch.optim
import torch.autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

torch.manual_seed(1)



# TODO: Remove when pytorch bugs/compatability are fixed
# import sys, warnings, traceback, torch
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
#     traceback.print_stack(sys._getframe(2))
# warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
# torch.utils.backcompat.broadcast_warning.enabled = True
# torch.utils.backcompat.keepdim_warning.enabled = True


"""
TODO: Later (NOT NOW!!) :
- Type definitions
- Documentation (properly formatted, meaning triple quote docs after class/fcns and sameline for variables)
"""

def _var(tensor, nograd=False):
    # Enable GPU optimization if possible
    return Variable(tensor, volatile=nograd).cuda() if torch.cuda.is_available() else Variable(tensor, volatile=nograd)



def _tensor(t_type, val=None):
    # Enable GPU optimization if possible
    tensor = getattr(torch.cuda, t_type) if torch.cuda.is_available() else getattr(torch, t_type)
    return tensor if val is None else tensor(val)


# TODO: IMP redo for new "same" gene structure ie no recursive updating for output
class NetworkNode(nn.Module):
    def __init__(self, gene, input_list=(), output_list=()):
        super(NetworkNode, self).__init__()

        self.G = gene
        # Input Connections
        self.input = []
        # Output connections
        self.output = []
        # self.input_tensor = None
        # self.tensor = None
        self.result = None

        if input_list: self.add_input(input_list)
        if output_list: self.add_output(output_list)


    def add_input(self, input_list):
        for node in input_list:
            if node not in self.input:
                self.input.append(node)
                node.add_output([self])
                # self._build_tensor()

    def remove_input(self, input_list):
        for node in input_list:
            if node in self.input:
                self.input.remove(node)
                node.remove_output([self])
                # self._build_tensor()

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
        super(NetworkInput, self).__init__(gene, output_list=output_list)
        self.node = lambda x: x

    def forward(self, input=None):
        if (input is not None) and (self.result is None):
            self.result = self.node(input)

        elif self.result is None:
            raise ValueError("There must be some input or a previously calculated result")

        return self.result.view(*self.G.d_out)

    # def _build_tensor(self):
    #     self.tensor = torch.Tensor()


class NetworkOutput(NetworkNode):
    """
    Required: "gid", "ntype", "input", "output"
    Optional: "activation"
    """

    def __init__(self, gene, input_list=()):
        super(NetworkOutput, self).__init__(gene, input_list=input_list)

        # If there's a sequence of activations, then apply each one to a column/feature dim
        if isinstance(self.G.activation, (list, tuple)):
            assert len(self.G.activation) == self.G.d_out[-1]
            self.act_list = []
            for a in self.G.activation:
                if isinstance(a, str): self.act_list.append(getattr(nn, a)())
                else: self.act_list.append(a)
            ndim = len(self.G.d_out) - 1
            self.act = lambda x: torch.cat((self.act_list[i](x.select(ndim, i)) for i in range(self.G.d_out[-1])), ndim)

        # Get the activation fcn if a string, or a given fcn
        else:
            if isinstance(self.G.activation, str):
                self.act = getattr(nn, self.G.activation)()
            else:
                self.act = self.G.activation


    def forward(self, input=None):
        if (input is not None) and (self.result is None):
            self.result = self.act(input)

        # Pull the input from previous network layers
        elif self.result is None:
            in_result = []

            # Apply a separate activation to each resulting input if applicable
            if self.G.in_activation:
                for i, n in enumerate(self.input):
                    in_result.append( self.G.in_activation[i](n()).type(_tensor("FloatTensor")) )

            else:
                for n in self.input:
                    in_result.append( n() )

            # Concatenate input along the lat dim
            self.result = self.act(torch.cat(in_result, in_result[0].dim() - 1))

        return self.result.view(*self.G.d_out)



    # def _build_tensor(self):
    #     # NOTE: Assumes dim n x m x k x ..., giving  n x m x sum(k) x ...
    #     if len(self.input) == 1:
    #         self.input_tensor = self.input[0].tensor
    #     else:
    #         self.input_tensor = torch.cat(self.input, dim=2)
    #     del self.tensor
    #
    #     if self.G.activation is not None:
    #         self.tensor = self.G.activation(self.input_tensor)
    #     else:
    #         self.tensor = self.input_tensor


class ConvLayer(NetworkNode):
    """
    Required: "gid", "ntype", "input", "output", "dim_filter"
    Optional: "activation", "stride"
    """

    def __init__(self, gene, input_list=(), output_list=()):
        super(ConvLayer, self).__init__(gene, input_list, output_list)


        # self.weights = tf.Variable(tf.random_uniform(self.gene["filtersize"], minval=-1.0, maxval=1.0))
        # self.biasvar = tf.Variable(tf.constant([0.1], shape=[self.gene["filtersize"][3]]))


    def forward(self, input=None):
        pass
    # def _build_tensor(self):
    #     # NOTE: Assumes dim n x m x k x ..., giving  n x m x sum(k) x ...
    #     if len(self.input) == 1:
    #         self.input_tensor = self.input[0].tensor
    #     else:
    #         self.input_tensor = tf.concat(axis=2, values=[n.tensor for n in self.input])
    #
    #     # NOTE: Assumes dim n x m x k x 1, reducing to n x m
    #     del self.tensor
    #     if "activation" in self.gene and self.gene["activation"] is not None:
    #         self.tensor = self.gene["activation"](self._conv(self.input_tensor, self.weights, self.biasvar))
    #     else:
    #         self.tensor = self._conv(self.input_tensor, self.weights, self.biasvar)
    #
    #     for node in self.output: node.add_input([self])
    #
    # def _conv(self, a, b, c):
    #     return tf.nn.conv2d(a, b, strides=[1, 1, 1, 1], padding='SAME') + c


class MaxPool(NetworkNode):
    # gene_space = {"nodetype": ["maxpool"],
    #               "input": ["conv", "recurrent", "ninput", "maxpool", "matmul"],
    #               "output": ["conv", "recurrent", "noutput", "maxpool", "matmul"]}
    """
    Required params: nodetype
    Optional: activation
    """

    def __init__(self, gene, input_list=(), output_list=()):
        super(MaxPool, self).__init__(gene, input_list, output_list)
        #
        # if input_list: self.add_input(input_list)
        # if output_list: self.add_output(output_list)

    def forward(self, input=None):
            pass
    # def _build_tensor(self):
    #     # NOTE: Assumes dim n x m x k x ..., giving  n x m x sum(k) x ...
    #     if len(self.input) == 1:
    #         self.input_tensor = self.input[0].tensor
    #     else:
    #         self.input_tensor = tf.concat(axis=2, values=[n.tensor for n in self.input])
    #
    #     # NOTE: Assumes dim n x m x k x 1, reducing to n x m
    #     del self.tensor
    #     if "activation" in self.gene and self.gene["activation"] is not None:
    #         self.tensor = self.gene["activation"](self._pool(self.input_tensor))
    #     else:
    #         self.tensor = self._pool(self.input_tensor)
    #
    #     for node in self.output: node.add_input([self])
    #
    # def _pool(self, a):
    #     return tf.nn.max_pool(a, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name="maxpool")


class FullyConnected(NetworkNode):
    """
    Required params: nodetype, activation, n_dim, m_dim
    Optional:
    """

    def __init__(self, gene, input_list=(), output_list=()):
        super(FullyConnected, self).__init__(gene, input_list, output_list)

        self.node = nn.Linear(self.G.d_in[-1], self.G.d_out[-1])

        # if self.G.activation:
        #     self.act = getattr(nn, self.G.activation)()
        # else:
        #     self.act = lambda x: x

        # If there's a sequence of activations, then apply each one to a column/feature dim
        if isinstance(self.G.activation, (list, tuple)):
            assert len(self.G.activation) == self.G.d_out[-1]
            self.act_list = []
            for a in self.G.activation:
                if isinstance(a, str): self.act_list.append(getattr(nn, a)())
                else: self.act_list.append(a)
            ndim = len(self.G.d_out) - 1

            # print(self.act_list)

            self.act = lambda x: torch.cat([self.act_list[i](col) for i, col in enumerate(torch.split(x, 1, ndim))], ndim)

        # Get the activation fcn if a string, or a given fcn
        else:
            if isinstance(self.G.activation, str):
                self.act = getattr(nn, self.G.activation)()
            else:
                self.act = self.G.activation


        if self.G.dropout:
            self.drop = nn.Dropout()
        else:
            self.drop = lambda x: x


    def forward(self, input=None):
        """

        :param input: dim = (n x d_in)
        :return: output dim = (n x d_out)
        """
        if (input is not None) and (self.result is None):

            self.result = self.act(self.drop(self.node(input.view(*self.G.d_in))))

        # Pull the input from previous network layers
        elif self.result is None:
            in_result = []
            for n in self.input:
                in_result.append( n() )

            # input = torch.cat(in_result, 0)

            # Concatenate input along the last dim
            self.result = self.act(self.drop(self.node(torch.cat(in_result, in_result[0].dim() - 1))))

        return self.result.view(*self.G.d_out)




class Embed(NetworkNode):

    def __init__(self, gene, input_list=(), output_list=()):
        super(Embed, self).__init__(gene, input_list, output_list)

        self.node = nn.Embedding(self.G.n_embed, self.G.d_out[-1], max_norm=self.G.max_norm)

        if self.G.dropout:
            self.drop = nn.Dropout()
        else:
            self.drop = lambda x: x

    def forward(self, input=None):
        """

        :param input: dim = (n x m)
        :return: output dim = (n x m x d_out)
        """
        if (input is not None) and (self.result is None):
            self.result = self.drop(self.node(input))

        # Pull the input from previous network layers
        elif self.result is None:
            in_result = []
            for n in self.input:
                in_result.append( n() )

            # Concatenate input along the last dim
            self.result = self.drop(self.node(torch.cat(in_result, in_result[0].dim()-1).type(_tensor("LongTensor"))))


        return self.result.view(*self.G.d_out)


class Recurrent(NetworkNode):

    types = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(self, gene, input_list=(), output_list=()):
        super(Recurrent, self).__init__(gene, input_list, output_list)

        dim = self.G.num_layers if not self.G.bidir else self.G.num_layers * 2

        self.hidden = _var(torch.rand(dim, self.G.d_batch, self.G.d_out[-1]))
        if self.G.ntype == "lstm":
            self.state = _var(torch.zeros(dim, self.G.d_batch, self.G.d_out[-1]))

        # # Enable GPU optimization
        # if torch.cuda.is_available():
        #     self.hidden = self.hidden.cuda()
        #     if self.G.ntype == "lstm": self.state = self.state.cuda()

        # if self.G.ntype == "rnn":
        #     self.node = Recurrent.types["rnn"](self.G.d_in[-1], self.G.d_out[-1], self.G.num_layers,
        #                              bidirectional=self.G.bidir)
        # else:
        self.node = Recurrent.types[self.G.ntype](self.G.d_in[-1], self.G.d_out[-1], self.G.num_layers,
                                     bidirectional=self.G.bidir)

    def forward(self, input=None):
        """

        :param input:
        :return: output dim = (n x m x d_hidden*dirs{1, 2})
        """

        self._repackage()
        if (input is not None) and (self.result is None):
            if self.G.ntype == "lstm":
                self.result, both = self.node(input, (self.hidden, self.state))
                self.hidden, self.state = both
            else:
                self.result, self.hidden = self.node(input, self.hidden)

        # Pull the input from previous network layers
        elif self.result is None:
            in_result = []
            for n in self.input:
                in_result.append( n() )

            # Concatenate input along the dim input_size
            if self.G.ntype == "lstm":
                self.result, both = self.node(torch.cat(in_result, 2), (self.hidden, self.state))
                self.hidden, self.state = both
            else:
                self.result, self.hidden = self.node(torch.cat(in_result, 2), self.hidden)

        return self.result.view(*self.G.d_out)


    def _repackage(self):
        """Wraps hidden states in new Variables, to detach them from their history."""
        self.hidden = _var(self.hidden.data)
        if self.G.ntype == "lstm":
            self.state = _var(self.state.data)

        # # Enable GPU optimization
        # if torch.cuda.is_available():
        #     self.hidden = self.hidden.cuda()
        #     if self.G.ntype == "lstm": self.state = self.state.cuda()

class NeuralNetwork(nn.Module):
    NodeTypes = {
                "nn_in": NetworkInput, "nn_out": NetworkOutput
                # , "conv": ConvLayer
                # , "maxpool": MaxPool, "avgpool": MaxPool,
                , "lin": FullyConnected, "embed": Embed
                , "rnn": Recurrent, "lstm": Recurrent, "gru": Recurrent
                }

    # NOTE: connections is a list of tuples of lists ([in], [out]) of gids, for each
    #       corresponding gene in the genes list
    # TODO: replace genes+connections with a genotype
    def __init__(self, genes, connections, inputs=(), outputs=()):
        super(NeuralNetwork, self).__init__()

        # self.genes = []
        self.nodes = {}
        self.input_nodes = []
        self.output_nodes = []

        # for g in genes:
        #     if isinstance(g, gn.GeneTypes) and g.gid in genome.allGenes:
        #         self.genes.append(g)
        #     elif isinstance(g, dict):
        #         self.genes.append(genome.new_gene(g))
        #     else:
        #         raise ValueError("Given argument is not a dict or gene")

        # self.genes = genes.values()

        # Make neural network layers (pytorch) modules for each gene
        for gid, g in genes.items():

            n = NeuralNetwork.NodeTypes[g.ntype](g)
            self.nodes[gid] = n
            self.add_module(g.ntype + str(gid), n)

            if gid in inputs: self.input_nodes.append(n)
            if gid in outputs: self.output_nodes.append(n)

        # Connect the modules
        for i, n in self.nodes.items():
            innodes = [self.nodes[j] for j in connections[i][0]]
            outnodes = [self.nodes[j] for j in connections[i][1]]
            n.add_input(innodes)
            n.add_output(outnodes)

            # Tie weights if applicable
            if hasattr(n.G, "tied") and n.G.tied is not None:
                assert self.nodes[n.G.tied].node.weight.size() == n.node.weight.size()
                n.node.weight = self.nodes[n.G.tied].node.weight


        # Default behavior is to put data in the input layers, get it out from the output layers
        if not any(self.input_nodes):
            self.input_nodes = list(n for n in self.nodes if n.G.ntype == "nn_in")
        if not any(self.output_nodes):
            self.output_nodes = list(n for n in self.nodes if n.G.ntype == "nn_out")

        # TODO: test - print (named) params, children module trees+parameters

    def forward(self, input, innodes=None, outnodes=None):
        """
        :param input: The input data OR assumes a list of inputs w/ same length as self.input_nodes.
        :return: The output from the network's output_nodes, in a list if there are multiple output nodes.
        """

        input_nodes = innodes if innodes else self.input_nodes
        output_nodes = outnodes if outnodes else self.output_nodes

        if len(input_nodes) == 1:
            input_nodes[0](input)
        else:
            for i, node in enumerate(input_nodes):
                node(input[i])

        if len(output_nodes) == 1:
            result = output_nodes[0]()
        else:
            result = []
            for node in output_nodes:
                result.append(node())

        for n in self.nodes.values():
            n.result = None

        return result


# TODO
# class HyperParamOpt:
#     """
#     Hyperparameter trainer.
#
#     :param params:
#     :returns:
#     """
#
#     def __init__(self, params):
#         pass

class NetworkRunner:
    errtypes = {
                "avg_err": (lambda x: torch.mean(x)),
                "perplexity": (lambda x: math.exp(x))
               }


    # TODO: remove assumption that data is in pytorch form
    def __init__(self, network , xtr, ytr, xte=None, yte=None, seq=True, dropout=False, format_fcn=None):

        self.network = network
        self.dropout = dropout
        self.seq = seq
        self.format_fcn = format_fcn

        if xte is not None:
            self.x_train = xtr
            self.y_train = ytr
            self.x_test = xte
            self.y_test = yte
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = self._make_test_data(xtr, ytr)
        

    def _make_test_data(self, data_x, data_y, amount=10):
        """
        amount: the amount to divide by (the percent of data that will be test)
        """
        num = len(data_x) // amount
        return data_x[num:], data_y[num:], data_x[:num], data_y[:num]

    def evaluate(self, data, data_y=None, criterion=None, batched=None, err=None):
        # Take out of training mode, into eval mode
        if self.dropout:
            self.network.eval()

        self.network.zero_grad()

        # Enable GPU optimization
        if torch.cuda.is_available():
            self.network.cuda()

        mult_in = None if len(self.network.input_nodes) <= 1 else len(self.network.input_nodes)
        mult_out = None if len(self.network.output_nodes) <= 1 else len(self.network.output_nodes)
        result = []
        
        # Batched input for when the sequence is too long for GPU to handle (??)
        if batched:       
            results = []
            # Multi outputs
            if mult_out:
                for j in range(mult_out): results.append([])          

            # Make batches
            len_data = len(data[0]) if mult_in else len(data)
            batches = [slice(i, i+batched) for i in range(0, len_data, batched) if i + batched < len_data]
            if batches[-1].stop < len_data:
                batches.append(slice(batches[-1].stop, len_data))

            
            print("{} batches with {} inputs, {} outputs, did: ".format(len(batches), mult_in, mult_out), end='')

            for i, b in enumerate(batches):
                x_ = [_var(x[b], True) for x in data] if mult_in else _var(data[b], True)

                # Multi outputs
                if mult_out:
                    multi = self.network(x_)
                    for j in range(mult_out): results[j].append(multi[j])
                else:
                    results.append(self.network(x_))

                print("{}".format(i), end=' ') 

            y_pred = [torch.cat(r, dim=0) for r in results] if mult_out else torch.cat(results, dim=0)
            print()

        # Not batching input
        else:
            x_ = [_var(x, True) for x in data] if mult_in else _var(data, True)
            y_pred = self.network(x_)
        

        result.append(y_pred)

        # Calculate loss
        if data_y is not None:
            if mult_out:
                y_ = [_var(x, True) for x in data_y]  
                loss = sum([criterion[i](y_pred[i], y_[i]).data[0] for i in range(len(y_))]) / mult_out
            else:
                y_ = _var(data_y, True)
                loss = criterion(y_pred, y_).data[0]

            result.append(loss)

        # Calculate error
        if err is not None:
            error = self._calc_err(err, loss=loss, y_pred=y_pred, y_=y_, len=mult_out)
            result.append(error)

        # Return to training mode
        if self.dropout:
            self.network.train()

        return result

    def train(self, epochs=50, batch_size=30, learn_rate=0.0005,
                      lossfcn="MSELoss", opt="SGD", err="avg_err", clipgrad=0.25, interval=50, l2=0.01, path=None, pretrained=None):

        if pretrained:
            self.network.load_state_dict(pretrained["model"])

        if self.dropout:
            self.network.train()

        # Enable GPU optimization
        if torch.cuda.is_available():
            self.network.cuda()

        # Use multiple losses if available
        if isinstance(lossfcn, (list, tuple)):
            criterion = [getattr(nn, l)() for l in lossfcn]
        else:
            criterion = getattr(nn, lossfcn)()

        # Set the optimizer
        optimizer = getattr(torch.optim, opt)(self.network.parameters(), lr=learn_rate, weight_decay=l2)
        

        # Init stats variables
        if pretrained:
            data = pretrained["data"]
            stats_per_epoch = pretrained["stats_per_epoch"]
            start = pretrained["starting_epoch"]
        else:
            data = {}
            for d in ["loss", "err"]:
                data[d] = []
            stats_per_epoch = []
            start = 0

        bsz = batch_size
        len_data = self.x_train[0].size(0) if len(self.network.input_nodes) > 1 else self.x_train.size(0)
        # Start training
        try:
            for e in range(start, epochs):
                # create batches (of random data points if not a sequence)
                if self.seq:
                    batches = [slice(i, i+bsz) for i in range(0, len_data - 1, bsz) \
                                if i + bsz < len_data]
                    if batches[-1].stop < len_data:
                        batches.append(slice(batches[-1].stop, len_data))
                else:
                    inds = np.random.permutation(range(len_data))
                    batches = [torch.LongTensor(inds[i:(i + bsz)]) for i in range(0, len(inds), bsz) \
                                if i + bsz < len(inds)]

                stats = {}
                stats["loss"] = [0]
                stats[err] = []
                for b in range(len(batches)):

                    # Multiple inputs
                    if len(self.network.input_nodes) > 1:
                        x_ = [_var(x[batches[b]]) for x in self.x_train]                
                    else:
                        x_ = _var(self.x_train[batches[b]])

                    # Run and evaluate the network
                    y_pred = self.network(x_)

                    # Evaulate loss over multiple outputs
                    if len(self.network.output_nodes) > 1:
                        y_ = [_var(x[batches[b]]) for x in self.y_train]  
                        loss = [criterion[i](y_pred[i], y_[i]) for i in range(len(y_))]
                    else:
                        y_ = _var(self.y_train[batches[b]])                 
                        loss = [criterion(y_pred, y_)]

                    # Save stats
                    stats["loss"][-1] += sum(x.data[0] for x in loss)

                    # Before the backward pass, use the optimizer object to zero all of the
                    # gradients for the variables it will update (which are the learnable weights
                    # of the model)
                    self.network.zero_grad()

                    # Backward pass: compute gradient of the loss with respect to model
                    # parameters
                    torch.autograd.backward(loss)
                    # torch.autograd.backward(loss, create_graph=True)
                    # loss.backward(retain_variables=True)


                    # Reduce exploding gradient problem
                    if clipgrad:
                        nn.utils.clip_grad_norm(self.network.parameters(), clipgrad)

                    # Update weights of the network
                    optimizer.step()

                    # Report and save performance for batch interval
                    if b % interval == 0:
                    # if b % interval == 0 and b != 0:
                        # Save stats
                        avgloss = stats["loss"][-1] / interval
                        stats["loss"][-1] = avgloss
                        stats[err].append(self._calc_err(err, loss=avgloss, y_pred=y_pred, y_=y_, len=len(loss)))

                        # if err == "perplexity":
                        #     stats[err].append(math.exp(stats["loss"][-1]))
                        # elif err == "avg_err":
                        #     stats[err].append(torch.mean(y_pred - y_).data[0])

                        # if path: self._save_data(path+"checkpoints/{}-{}-".format(e, b), stats=stats)


                        print("Epoch: {}\t Batch: {}/{}\t Loss: {}\t {}: {}\t".format(
                                e, b, len(batches), avgloss, err, stats[err][-1]))

                        stats["loss"].append(0)


                # Run on test data
                y_pred, loss, error = self.evaluate(self.x_test, self.y_test, criterion, batched=500, err=err)
                # y_ =  [_var(y) for y in self.y_test] if len(self.network.output_nodes) > 1 else _var(self.y_test)
                # error = self._calc_err(err, loss=loss, y_pred=y_pred, y_=y_, 
                #                         len=len(self.network.output_nodes))
                # if err == "perplexity":
                #     error = math.exp(loss)
                # elif err == "avg_err":
                #     error = torch.mean(y_pred - _var(self.y_test).squeeze()).data[0]

                # Save stats for epoch
                stats_per_epoch.append(stats)
                # data["output"].append(y_pred)
                data["loss"].append(loss)
                data["err"].append(error)

                if path: self._save_data(path+"checkpoints/epoch_{}-".format(e), data=data, stats_per_epoch=stats_per_epoch, model=self.network)

                # Report on test data
                print("TEST DATA -- \t Epoch: {}\t Loss: {}\t {}: {}\t".format(e, loss, err, error))


            # Save all to file after all epochs
            print("Finished {} epochs, saving to {}".format(epochs, path))
            if path: self._save_data(path, model=self.network, stats=data, stats_per_epoch=stats_per_epoch)

        except KeyboardInterrupt:
            print("Quitting from interrupt")

            stats["loss"][-1] = stats["loss"][-1] / (interval if b % interval == 0 else b % interval)
            stats[err].append(self._calc_err(err, loss=stats["loss"][-1], y_pred=y_pred, y_=y_, 
                                len=len(self.network.output_nodes)))

            stats_per_epoch.append(stats)
            
            print("Saving to {}".format(path))
            if path: self._save_data(path+"interrupt/", model=self.network, stats=data, stats_per_epoch=stats_per_epoch)
            print("Finished saving")


        
    def _calc_err(self, errtype, **kwargs):
        if errtype == "perplexity":
            result = math.exp(kwargs["loss"])

        elif errtype == "avg_err":
            if "len" in kwargs and kwargs["len"] and kwargs["len"] > 1:
                if self.format_fcn:
                    y_pred = [self.format_fcn[i](kwargs["y_pred"][i]) for i in range(kwargs["len"])]
                else: 
                    y_pred = kwargs["y_pred"]

                errs = [torch.mean(y_pred[i].sub_(kwargs["y_"][i]).type(_tensor("FloatTensor"))).data[0] for i in range(kwargs["len"])]
                result = sum(errs) / kwargs["len"]

            else:
                result = torch.mean((kwargs["y_pred"] - kwargs["y_"]).type(_tensor("FloatTensor"))).data[0]

        return result



    def _save_data(self, path, **kwargs):
        
        for key, value in kwargs.items():
            if not os.path.exists(path[0:path.rindex("/")]):
                os.makedirs(path[0:path.rindex("/")])
            with open(path + key, 'wb') as file:
                if key == "model":
                    torch.save(value.state_dict(), file)
                else:
                    pickle.dump(value, file)
                print("Saved {}".format(key))
