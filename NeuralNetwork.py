#! python3.6

"""
Author: Bhavana Jonnalagadda, 2017
"""

import math
import pickle
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


"""
TODO: Later (NOT NOW!!) :
- Type definitions
- Documentation (properly formatted, meaning triple quote docs after class/fcns and sameline for variables)
"""



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

        return self.result

    # def _build_tensor(self):
    #     self.tensor = torch.Tensor()


class NetworkOutput(NetworkNode):
    """
    Required: "gid", "ntype", "input", "output"
    Optional: "activation"
    """

    def __init__(self, gene, input_list=()):
        super(NetworkOutput, self).__init__(gene, input_list=input_list)

        # self.biasvar = tf.Variable(tf.constant([1.0]))
        # self.biasvar2 = tf.Variable(tf.constant([1.0]))
        if self.G.activation is None:
            self.node = lambda x: x
        else:
            self.node = getattr(F, self.G.activation)


    def forward(self, input=None):
        if (input is not None) and (self.result is None):
            self.result = self.node(input).view(*self.G.d_out)

        # Pull the input from previous network layers
        elif self.result is None:
            in_result = []
            for n in self.input:
                in_result.append( n() )

            # Concatenate input along the 3rd dim
            # TODO: make concatenation on a given dim or default last dim
            self.result = self.node(torch.cat(in_result, 2)).view(*self.G.d_out)

        return self.result



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
    gene_space = {"nodetype": ["matmul"],
                  "input": ["conv", "recurrent", "ninput", "maxpool", "matmul"],
                  "output": ["conv", "recurrent", "noutput", "maxpool", "matmul"]}
    """
    Required params: nodetype, activation, n_dim, m_dim
    Optional:
    """

    def __init__(self, gene, input_list=(), output_list=()):
        super(FullyConnected, self).__init__(gene, input_list, output_list)

        self.node = nn.Linear(self.G.d_in[-1], self.G.d_out)

        if self.G.activation:
            self.act = getattr(nn, self.G.activation)()
        else:
            self.act = lambda x: x

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

            self.result = self.act(self.drop(self.node(input.view(*self.G.d_in)))).view(input.size()[:-1], self.G.d_out)

        # Pull the input from previous network layers
        elif self.result is None:
            in_result = []
            for n in self.input:
                in_result.append(n())

            input = torch.cat(in_result, 0)
            # Concatenate input along the dim "n"
            self.result = self.act(self.drop(self.node(input.view(*self.G.d_in)))).view(*list(input.size()[:-1]), self.G.d_out)

        return self.result




class Embed(NetworkNode):

    def __init__(self, gene, input_list=(), output_list=()):
        super(Embed, self).__init__(gene, input_list, output_list)

        self.node = nn.Embedding(self.G.d_embed, self.G.d_out)

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

            # Concatenate input along the dim "m"
            self.result = self.drop(self.node(torch.cat(in_result, 1)))


        return self.result


class Recurrent(NetworkNode):

    types = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(self, gene, input_list=(), output_list=()):
        super(Recurrent, self).__init__(gene, input_list, output_list)

        dim = self.G.num_layers if not self.G.bidir else self.G.num_layers * 2

        self.hidden = Variable(torch.zeros(dim, self.G.d_batch, self.G.d_hidden))
        if self.G.ntype == "lstm":
            self.state = Variable(torch.zeros(dim, self.G.d_batch, self.G.d_hidden))

        if self.G.ntype == "rnn":
            self.node = Recurrent.types["rnn"](self.G.d_in, self.G.d_hidden, self.G.num_layers, self.G.nonlin,
                                     bidirectional=self.G.bidir)
        else:
            self.node = Recurrent.types[self.G.ntype](self.G.d_in, self.G.d_hidden, self.G.num_layers,
                                     bidirectional=self.G.bidir)

    def forward(self, input=None):
        """

        :param input:
        :return: output dim = (n x m x d_hidden*dirs{1, 2})
        """
        if (input is not None) and (self.result is None):
            if self.G.ntype == "lstm":
                self.result, self.hidden, self.state = self.node(input, self.hidden, self.state)
            else:
                self.result, self.hidden = self.node(input, self.hidden)

        # Pull the input from previous network layers
        elif self.result is None:
            in_result = []
            for n in self.input:
                in_result.append( n() )

            # Concatenate input along the dim input_size
            if self.G.ntype == "lstm":
                self.result, self.hidden, self.state = self.node(torch.cat(in_result, 2), self.hidden, self.state)
            else:
                self.result, self.hidden = self.node(torch.cat(in_result, 2), self.hidden)

        return self.result



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
    def __init__(self, genes, connections, genome, inputs=(), outputs=()):
        super(NeuralNetwork, self).__init__()

        self.genes = []
        self.nodes = nn.ModuleList()
        self.input_nodes = []
        self.output_nodes = []

        for g in genes:
            if isinstance(g, gn.GeneTypes) and g.gid in genome.allGenes:
                self.genes.append(g)
            elif isinstance(g, dict):
                self.genes.append(genome.new_gene(g))
            else:
                raise ValueError("Given argument is not a dict or gene")

        # Make neural network layers (pytorch) modules for each gene
        for g in genes:

            n = NeuralNetwork.NodeTypes[g.ntype](g)
            self.nodes.append(n)

            if g.gid in inputs: self.input_nodes.append(n)
            if g.gid in outputs: self.output_nodes.append(n)

        # Connect the modules
        for (i, n) in enumerate(self.nodes):
            innodes = [self.nodes[j] for j in connections[i][0]]
            outnodes = [self.nodes[j] for j in connections[i][1]]
            n.add_input(innodes)
            n.add_output(outnodes)

        # Default behavior is to put data in the input layers, get it out from the output layers
        if not any(self.input_nodes):
            self.input_nodes = list(n for n in self.nodes if n.G.ntype == "nn_in")
        if not any(self.output_nodes):
            self.output_nodes = list(n for n in self.nodes if n.G.ntype == "nn_out")

        # TODO: test - print (named) params, children module trees+parameters

    def forward(self, input):
        """
        :param input: The input data OR assumes a list of inputs w/ same length as self.input_nodes.
        :return: The output from the network's output_nodes, in a list if there are multiple output nodes.
        """
        if len(self.input_nodes) == 1:
            self.input_nodes[0](input)
        else:
            for i, node in enumerate(self.input_nodes):
                node(input[i])

        if len(self.output_nodes) == 1:
            result = self.output_nodes[0]()
        else:
            result = []
            for node in self.output_nodes:
                result.append(node())

        for n in self.nodes:
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


    # TODO: make test data optional
    def __init__(self, network , xtr, ytr, xte, yte, seq=False):

        self.network = network
        self.x_train = xtr
        self.y_train = ytr
        self.x_test = xte
        self.y_test = yte
        self.seq = seq

    def evaluate(self, data, data_y=None, criterion=None):
        self.network.eval()
        result = self.network(data)
        if data_y:
            loss = criterion(result, data_y)
        self.network.train()

        if data_y:
            return result, loss.data[0]
        else:
            return result

    def train(self, epochs=50, batch_size=30, learn_rate=0.0005,
                      lossfcn="MSELoss", opt="SGD", err="avg_err", clipgrad=0.25, interval=50, path=None):

        self.network.train()

        # Enable GPU optimization
        # if torch.cuda.is_available():
        #     self.network.cuda()

        # Set hyperparameters
        criterion = getattr(nn, lossfcn)()
        optimizer = getattr(torch.optim, opt)(self.network.parameters(), lr=learn_rate)
        b = batch_size

        data = {}
        for d in ['output', "loss", "err"]:
            data[d] = []
        stats_per_epoch = []

        try:
            for e in range(epochs):
                # create batches (of random data points if not a sequence)
                if self.seq:
                    batches = [slice(i, i+b) for i in range(0, self.x_train.size(0) - 1, b) if i + b < len(self.x_train)]
                else:
                    inds = np.random.permutation(range(len(self.x_train)))
                    batches = [torch.LongTensor(inds[i:(i + b)]) for i in range(0, len(inds), b) if i + b < len(inds)]

                stats = {}
                stats["loss"] = [0]
                stats[err] = []
                for i in range(len(batches)):
                    x_ = Variable(self.x_train[batches[i]])
                    y_ = Variable(self.y_train[batches[i]]).view(-1)

                    # Run and evaluate the network
                    y_pred = self.network(x_)
                    loss = criterion(y_pred, y_)

                    # Save stats
                    stats["loss"][-1] += loss.data[0]

                    # Before the backward pass, use the optimizer object to zero all of the
                    # gradients for the variables it will update (which are the learnable weights
                    # of the model)
                    self.network.zero_grad()

                    # Backward pass: compute gradient of the loss with respect to model
                    # parameters
                    loss.backward(retain_variables=True)

                    # Reduce exploding gradient problem
                    if clipgrad:
                        nn.utils.clip_grad_norm(self.network.parameters(), clipgrad)

                    # Update weights of the network
                    optimizer.step()

                    # Report and save performance
                    if i % interval == 0:
                        # Save stats
                        stats["loss"][-1] = stats["loss"][-1] / interval
                        if err == "perplexity":
                            stats[err].append(math.exp(stats["loss"][-1]))
                        elif err == "avg_err":
                            stats[err].append(torch.mean(y_pred - y_))


                        print("Epoch: {}\t Batch: {}/{}\t Loss: {}\t {}: {}\t".format(
                                e, i, len(batches), stats["loss"][-1], err, stats[err][-1]))

                        stats["loss"].append(0)

            # Save stats
            y_pred, loss = self.evaluate(self.x_test, self.y_test, criterion)
            if err == "perplexity":
                error = math.exp(loss)
            elif err == "avg_err":
                error = torch.mean(y_pred - self.y_test)
            stats_per_epoch.append(stats)
            data["output"].append(y_pred)
            data["loss"].append(loss)
            data["err"].append(error)

            # Report on test data
            print("Epoch: {}\t Loss: {}\t {}: {}\t".format(e, loss, err, error))

        except KeyboardInterrupt:
            print("Quitting from interrupt")

            stats["loss"][-1] = stats["loss"][-1] / (interval if i % interval == 0 else i % interval)
            if err == "perplexity":
                            stats[err].append(math.exp(stats["loss"][-1]))
                        elif err == "avg_err":
                            stats[err].append(torch.mean(y_pred - y_))
            stats_per_epoch.append(stats)
            
            if path: self._save_data(path, model=self.network, data=data, stats_per_epoch=stats_per_epoch)
            print("Finished saving")

        # Save all to file
        if path: self._save_data(path, model=self.network, data=data, stats_per_epoch=stats_per_epoch)




    def _save_data(self, path, **kwargs):
        print("Saving to {}".format(path))
        for key, value in kwargs.items():
            with open(path + key, 'wb') as file:
                if key == "model":
                    torch.save(value.state_dict(), file)
                else:
                    pickle.dump(value, file)
                print("Saved {}".format(key))
