#! python3.6
"""
Creates an artificial neural network by initializing layers using parameters 
from `NNGenome`. Built on top of pytorch, providing a convenient abstraction.

Author: Bhavana Jonnalagadda, 2017
"""
import os
import math
import pickle
import random
import numpy as np
import torch, torch.optim
import torch.autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import NNGenome as gn

torch.manual_seed(1)


"""
Helper functions
"""
def _var(tensor, nograd=False):
    # Enable GPU optimization if possible
    return Variable(tensor, volatile=nograd).cuda() if torch.cuda.is_available() else Variable(tensor, volatile=nograd)

def _tensor(t_type, val=None):
    # Enable GPU optimization if possible
    tensor = getattr(torch.cuda, t_type) if torch.cuda.is_available() else getattr(torch, t_type)
    return tensor if val is None else tensor(val)


class NetworkNode(nn.Module):
    """Base class for all neural network layers/nodes.

    For now, just provides a centralized way to add and remove connections
    between layers and to call the needed pytorch `nn.Module` initialization. 

    Note:
        All derived classes must define `self.forward` which performs the actual
        calculation and stores in `result`.

    Attributes:
        G: Stores the gene created by `NNGenome` which has all the needed 
                info for this layer node.
        result: When the network is run, the output of this layer is 
                stored here so that it won't be run multiple times
                over the course of a network's single evaulation;
                it is deleted after the network is finished running.
        node: The actual pytorch module that does the work for the 
                specific layer; defined in derived classes.
    """
    def __init__(self, gene, input_list=(), output_list=()):
        super(NetworkNode, self).__init__()

        self.G = gene
        # Input Connections
        self.input = []
        # Output connections
        self.output = []
        self.result = None

        if input_list: self.add_input(input_list)
        if output_list: self.add_output(output_list)


    def add_input(self, input_list):
        for node in input_list:
            if node not in self.input:
                self.input.append(node)
                node.add_output([self])

    def remove_input(self, input_list):
        for node in input_list:
            if node in self.input:
                self.input.remove(node)
                node.remove_output([self])

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
    """
    Serves as a simple wrapper around input data.
    """
    def __init__(self, gene, output_list=()):
        super(NetworkInput, self).__init__(gene, output_list=output_list)
        self.node = lambda x: x

    def forward(self, input=None):
        """ Reshapes the data at most. """
        if (input is not None) and (self.result is None):
            self.result = self.node(input)

        elif self.result is None:
            raise ValueError("There must be some input or a previously calculated result")

        return self.result.view(*self.G.d_out)


class NetworkOutput(NetworkNode):
    """Wrapper around the output, and can run activations.

    Args:
        G.activation (str, fcn): runs a single activation over all
            data given to this layer.
        G.activation (list(str), list(fcn)): length of list must be the 
            same as size of last dim. Runs each activation separately
            on each feature in last dim.
    """

    def __init__(self, gene, input_list=()):
        super(NetworkOutput, self).__init__(gene, input_list=input_list)

        # If there's a sequence of activations, then apply each one to a column/feature dim
        if isinstance(self.G.activation, (list, tuple)):
            assert len(self.G.activation) == self.G.d_out[-1]
            self.act_list = []
            for a in self.G.activation:
                # If str, apply the str
                if isinstance(a, str): self.act_list.append(getattr(nn, a)())
                else: self.act_list.append(a)
            ndim = len(self.G.d_out) - 1
            # Run the activations on each column and then concatentate back together
            self.act = lambda x: torch.cat((self.act_list[i](x.select(ndim, i)) for i in range(self.G.d_out[-1])), ndim)

        # Get the activation fcn if a string, or a given fcn
        else:
            if isinstance(self.G.activation, str):
                self.act = getattr(nn, self.G.activation)()
            else:
                self.act = self.G.activation


    def forward(self, input=None):
        """ Perform activation if applicable and reshape. """
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


class ConvLayer(NetworkNode):
    """
    Not implemented
    """

    def __init__(self, gene, input_list=(), output_list=()):
        super(ConvLayer, self).__init__(gene, input_list, output_list)

    def forward(self, input=None):
        pass


class MaxPool(NetworkNode):
    """
    Not implemented
    """

    def __init__(self, gene, input_list=(), output_list=()):
        super(MaxPool, self).__init__(gene, input_list, output_list)

    def forward(self, input=None):
            pass


class FullyConnected(NetworkNode):
    """Performs a linear transformation to input with a matrix of weights and a bias.

    Weight matrix dim = (d_in, d_out)

    Args:
        G.activation (str, fcn): runs a single activation over all
            data given to this layer.
        G.activation (list(str), list(fcn)): length of list must be the 
            same as size of last dim. Runs each activation separately
            on each feature in last dim.
    """

    def __init__(self, gene, input_list=(), output_list=()):
        super(FullyConnected, self).__init__(gene, input_list, output_list)

        self.node = nn.Linear(self.G.d_in[-1], self.G.d_out[-1])

        # If there's a sequence of activations, then apply each one to a column/feature dim
        if isinstance(self.G.activation, (list, tuple)):
            assert len(self.G.activation) == self.G.d_out[-1]
            self.act_list = []
            for a in self.G.activation:
                # If str, apply the str
                if isinstance(a, str): self.act_list.append(getattr(nn, a)())
                else: self.act_list.append(a)
            ndim = len(self.G.d_out) - 1

            # Run the activations on each column and then concatentate back together
            self.act = lambda x: torch.cat([self.act_list[i](col) for i, col in enumerate(torch.split(x, 1, ndim))], ndim)

        # Get the activation fcn if a string, or a given fcn
        else:
            if isinstance(self.G.activation, str):
                self.act = getattr(nn, self.G.activation)()
            else:
                self.act = self.G.activation

        # Do dropout on weights if applicable
        if self.G.dropout:
            self.drop = nn.Dropout()
        else:
            self.drop = lambda x: x


    def forward(self, input=None):
        """Multiplies input with weights. 

        Does dropout/activation if applicable. Output dim = (n x d_out). Concatentates
        inputs along the last dim

        Args:
            input: dim = (n x d_in)
        """
        if (input is not None) and (self.result is None):
            self.result = self.act(self.drop(self.node(input.view(*self.G.d_in))))

        # Pull the input from previous network layers
        elif self.result is None:
            in_result = []
            for n in self.input:
                in_result.append( n() )

            # Concatenate input along the last dim
            self.result = self.act(self.drop(self.node(torch.cat(in_result, in_result[0].dim() - 1))))

        return self.result.view(*self.G.d_out)




class Embed(NetworkNode):
    """Takes a fixed-size class and gives a vectored representation (embedding) of the class label.

    Weight matrix dim = (size of class, size of embedding output). 
    Output dim = (n x m x d_out). 
    Input dim = (n x m). Concatenates input along the last dim
    """
    def __init__(self, gene, input_list=(), output_list=()):
        super(Embed, self).__init__(gene, input_list, output_list)

        self.node = nn.Embedding(self.G.n_embed, self.G.d_out[-1], max_norm=self.G.max_norm)

        if self.G.dropout:
            self.drop = nn.Dropout()
        else:
            self.drop = lambda x: x

    def forward(self, input=None):
        """Takes a label (integer) and gives a embedding vector.

        Args:
            input: dim = (n x m)
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
    """A recurrent layer. (see http://pytorch.org/docs/master/nn.html#recurrent-layers)

    Can be bidirectional (num_directions = 1, 2) i.e sequence is given both forwards and
    backwards to the layer. LSTMs have an additional hidden state.
    Hidden state dim = (num_layers*num_directions x batch x hidden_size). 
    Output dim = (n x batch x d_hidden*num_directions). 
    Input dim = (n x batch x input_size).
    """
    types = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(self, gene, input_list=(), output_list=()):
        super(Recurrent, self).__init__(gene, input_list, output_list)

        dim = self.G.num_layers if not self.G.bidir else self.G.num_layers * 2

        self.hidden = _var(torch.rand(dim, self.G.d_batch, self.G.d_out[-1]))
        if self.G.ntype == "lstm":
            self.state = _var(torch.zeros(dim, self.G.d_batch, self.G.d_out[-1]))

        self.node = Recurrent.types[self.G.ntype](self.G.d_in[-1], self.G.d_out[-1], self.G.num_layers,
                                     bidirectional=self.G.bidir)

    def forward(self, input=None):
        """Takes 3d input and gives 3d output from recurrent layers.
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



class NeuralNetwork(nn.Module):
    """
    Attributes:
    """
    NodeTypes = {
                "nn_in": NetworkInput, "nn_out": NetworkOutput
                # , "conv": ConvLayer
                # , "maxpool": MaxPool, "avgpool": MaxPool,
                , "lin": FullyConnected, "embed": Embed
                , "rnn": Recurrent, "lstm": Recurrent, "gru": Recurrent
                }

    def __init__(self, genes, connections, inputs=(), outputs=()):
        """
        Args:
            genes ({int: `GeneTypes`}): Dict of genes, should be the
                initialized `NNGenome.allGenes`.
            connections ({int: ([int], [int])}): Dict with key= gid, 
                value= tuple of lists ([in], [out]) of gids, for each 
                corresponding gene in the genes list, giving the input and
                output for that gid key
            inputs (optional): Where the given data in `self.forward` 
                is expected to go.
            outputs (optional): List of output. Loss during training is 
                applied to these.
        """
        super(NeuralNetwork, self).__init__()

        # self.genes = []
        self.nodes = {}
        self.input_nodes = []
        self.output_nodes = []

        self.format_fcn = None

        # Make neural network layers (pytorch) modules for each gene
        for gid, g in genes.items():

            n = NeuralNetwork.NodeTypes[g.ntype](g)
            self.nodes[gid] = n
            self.add_module(g.ntype + str(gid), n)

            # Save the activations for use with NetworkRunner
            if isinstance(g, gn.NNOutputGene) and g.in_activation:
                self.format_fcn = g.in_activation

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
            self.input_nodes = list(n for n in self.nodes.values() if n.G.ntype == "nn_in")
        if not any(self.output_nodes):
            self.output_nodes = list(n for n in self.nodes.values() if n.G.ntype == "nn_out")

    def forward(self, input, innodes=None, outnodes=None):
        """
        Args:
            input: The input data OR assumes a list of inputs w/ same length as self.input_nodes.
            innodes (optional):
            outnodes (optional):

        Returns:
            result: The output from the network's `output_nodes`, in a list if there are multiple output nodes.
        """

        # Set input/output
        input_nodes = innodes if innodes else self.input_nodes
        output_nodes = outnodes if outnodes else self.output_nodes

        # Feed input into the network
        if len(input_nodes) == 1:
            input_nodes[0](input)
        else:
            for i, node in enumerate(input_nodes):
                node(input[i])

        # Get output
        if len(output_nodes) == 1:
            result = output_nodes[0]()
        else:
            result = []
            for node in output_nodes:
                result.append(node())

        # Clear the calculated results stored in the network layers
        for n in self.nodes.values():
            n.result = None

        return result
