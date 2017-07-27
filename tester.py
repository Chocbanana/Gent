#! python3.6

import numpy as np
# import tensorflow as tf
import NeuralNetwork as ann
import NNGenome as gn

import data

import torch
import torch.nn as nn



# Change colormap for plotting
# plt.rcParams['image.cmap'] = 'bone'

"""
Load data
"""
datum = torch.load("data/ptb_torch")


"""
Initialize and run
"""
# TODO: get n_embeds from corpus.dictionary size; change dim code for lin, out
nlabels = len(datum["dictionary"])
nbatch = 10
nencode = 200
nhidden = 220
geneseq = [ {
            "gid": 0, "ntype": "nn_in", "d_in": -1, "d_out": -1
            },
            {
            "gid": 1, "ntype": "embed", "d_in": (-1, nbatch), "d_out": nencode
            , "d_embed": nlabels, "dropout": True
            },
            {
            "gid": 2, "ntype": "gru", "d_in": nencode, "d_out": nhidden
            , "d_hidden": nhidden, "d_batch": nbatch, "num_layers": 2
            },
            {
            "gid": 3, "ntype": "lin", "d_in": [-1, nhidden], "d_out": nlabels
            , "dropout": True
            },
            {
            "gid": 4, "ntype": "nn_out", "d_in": nlabels, "d_out": (-1, nlabels)
            }
        ]

gen = gn.NNGenome(geneseq)
connections = [([], [1]), ([0], [2]), ([1], [3]), ([2], [4]), ([3], []),]
network = ann.NeuralNetwork(gen.allGenes.values(), connections, gen)

# print(network.state_dict())

runner = ann.NetworkRunner(network, datum["xtr"], datum["ytr"], datum["xte"], datum["yte"], seq=True)
runner.train(batch_size=35, lossfcn="CrossEntropyLoss", err="perplexity", interval=2, path="logs/test/")
# print(gen)

# init_genes = [{"geneID": 0, "nodetype": "ninput", "input": [], "output": [1]},
#         {"geneID": 1,
#          "nodetype": "conv", "input": [0], "output": [2],
#          "filtersize": [5, 5, 1, 32],
#          "activation": tf.nn.relu
#         },
#         {"geneID": 2,
#          "nodetype": "maxpool", "input": [1], "output": [3]
#         },
#         {"geneID": 3,
#          "nodetype": "conv", "input": [2], "output": [4],
#          "filtersize": [5, 5, 32, 64],
#          "activation": tf.nn.relu
#         },
#         {"geneID": 4,
#          "nodetype": "maxpool", "input": [3], "output": [5]
#         },
#         {"geneID": 5,
#          "nodetype": "matmul", "input": [4], "output": [6],
#          # "activation": lambda x: tf.nn.dropout(tf.nn.relu(x), tf.placeholder(tf.float32)),
#          "activation": tf.nn.relu,
#          "n_dim": 7 * 7 * 64,
#          "m_dim": 1024
#         },
#         {"geneID": 6,
#          "nodetype": "matmul", "input": [5], "output": [7],
#          "n_dim": 1024,
#          "m_dim": 10
#         },
#         {"geneID": 7,
#          "nodetype": "noutput", "input": [6], "output": []
#         }]
#
# McIntosh .... >_>
# init_genes = [{"geneID": 0, "nodetype": "ninput", "input": [], "output": [1]},
#         {"geneID": 1, "nodetype": "conv", "channels": [1, 8], "activation": tf.nn.relu6, "filtersize": 16, "input": [0], "output": [2]},
#         {"geneID": 2, "nodetype": "conv", "channels": [8, 16], "activation": tf.nn.relu6, "filtersize": 16, "input": [1], "output": [3]},
#         {"geneID": 3, "nodetype": "noutput", "activation": tf.nn.relu6, "input": [2], "output": []}]

# basedir = "/Users/Bhavana/OneDrive/Documents/School/15-16/Winter/CNS 186/project/code/tf_logs"
# nn = ann.NeuralNetwork(init_genes)
# runner = ann.NetworkRunner(basedir, x_train_both, y_train_both, x_test1, y_test1)
# print(nn)
# stats1 = runner.train_network(nn, subdir="/m_reimpl", epochs=100, batch_size=10, learn_rate=0.0001)



