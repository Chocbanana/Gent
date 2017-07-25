#! python3.6

import numpy as np
# import tensorflow as tf
import NeuralNetwork as ann




# Change colormap for plotting
# plt.rcParams['image.cmap'] = 'bone'

# def load_and_format_data():
#     """
#     Load data
#     """
#     # Repeats: 1800 x 4 file, total time = 30s at 0.0167221s intervals
#     data1 = np.loadtxt('../data/Repeats.txt')
#     data1[:, 3] *= 0.0167221
#
#     # NoRepeats: 8901 x 4 + 26500 x 3 file, total time = 592s at 0.0167221s intervals
#     raw2 = np.loadtxt('../data/NoRepeats.txt', usecols=(0, 1, 2))
#     spikes = np.loadtxt('../data/NoRepeats_missingcolrows_removed.txt', usecols=(3,) )
#
#     # Convert the spike times into spike rates at the appropriate frame
#     rates = np.zeros((35401, 1))
#     for val in spikes:
#         ind = next((ind for ind, x in enumerate(raw2[:, 0]) if x > val), 0)
#         if ind == 0:
#             break
#         rates[ind-1] += 1
#
#     data2 = np.hstack((raw2[:, 1:3], (rates)))
#     data2 = data2[:-1, :]
#
#     """
#     Format data
#     """
#     # both 1800 and 35400 must be divisble by timelength
#     # 40 * 0.0167221s = 668.9ms
#     time_length = 60
#
#     x_test1 = data1[np.newaxis, :, 1:3, np.newaxis]
#     y_test1 = data1[np.newaxis, :, 3]
#
#     x_train1 = np.reshape(data1[:, 1:3], (-1, time_length, 2, 1))
#     y_train1 = np.reshape(data1[:, 3], (-1, time_length))
#     # Double the training size with the time chunks shifted
#     x2 = np.reshape(data1[int(time_length/2):int(-time_length/2), 1:3], (-1, time_length, 2, 1))
#     y2 = np.reshape(data1[int(time_length/2):int(-time_length/2), 3], (-1, time_length))
#     x_train1 = np.concatenate((x_train1, x2), 0)
#     y_train1 = np.concatenate((y_train1, y2), 0)
#
#     x_test2 = data2[np.newaxis, :, 0:2, np.newaxis]
#     y_test2 = data2[np.newaxis, :, 2]
#
#     x_train2 = np.reshape(data2[:, 0:2], (-1, time_length, 2, 1))
#     y_train2 = np.reshape(data2[:, 2], (-1, time_length))
#     # Double the training size with the time chunks shifted
#     x2 = np.reshape(data2[int(time_length/2):int(-time_length/2), 0:2], (-1, time_length, 2, 1))
#     y2 = np.reshape(data2[int(time_length/2):int(-time_length/2), 2], (-1, time_length))
#     x_train2 = np.concatenate((x_train2, x2), 0)
#     y_train2 = np.concatenate((y_train2, y2), 0)
#
#     x_train_both = np.concatenate((x_train1, x_train2), 0)
#     y_train_both = np.concatenate((y_train1, y_train2), 0)
#
#     np.savez("../data/formatted_spiking", x_train1=x_train1, y_train1=y_train1, x_train2=x_train2,
#         y_train2=y_train2, x_train_both=x_train_both, y_train_both=y_train_both,
#         y_test1=y_test1, x_test1=x_test1)
#     print("Saved data")
#
#     return x_train1, y_train1, x_train2, y_train2, x_train_both, y_train_both, x_test1, y_test1
#
#
# try:
#     # formatted_data = np.load("../data/formatted_spiking.npz")
#     formatted_data = np.load("../data/formatted.npz")
#     dictget = lambda k: formatted_data[k]
#     x_train1, y_train1, x_train2, y_train2, x_train_both, y_train_both, x_test1, y_test1 = map(dictget, \
#                                                     ("x_train1", "y_train1", "x_train2", \
#                                                      "y_train2", "x_train_both", "y_train_both", \
#                                                      "x_test1", "y_test1"))
# except (IOError, ValueError):
#     x_train1, y_train1, x_train2, y_train2, x_train_both, y_train_both, x_test1, y_test1 = load_and_format_data()
#
#
# # TODO: notes:
# #     - things changed: loss fcn (simply not working), data related nums (filter size, training data shape)
# #     - things assumed: learning rate, regularization coefficients, batch size
# #     - improvements: only the order of magnitude diff b/w regularization coefficients, but not sure what theirs were
# #                     - interesting to note that error: 0.373215  corr: 0.895872, but for them their corr (ie
# #                        performance) never reached past 0.7
# #                     - shows that the methods for judging performance do not translate across datasets?
#
# # epoch: 99  loss: 22.380592 error: 1.267139  corr: 0.863753
# #
# #



"""
Initialize and run
"""
init_genes = [{"geneID": 0, "nodetype": "ninput", "input": [], "output": [1]},
        {"geneID": 1, 
         "nodetype": "conv", "input": [0], "output": [2],
         "filtersize": [5, 5, 1, 32], 
         "activation": tf.nn.relu
        },
        {"geneID": 2, 
         "nodetype": "maxpool", "input": [1], "output": [3]
        },
        {"geneID": 3, 
         "nodetype": "conv", "input": [2], "output": [4],
         "filtersize": [5, 5, 32, 64], 
         "activation": tf.nn.relu
        },
        {"geneID": 4, 
         "nodetype": "maxpool", "input": [3], "output": [5]
        },
        {"geneID": 5, 
         "nodetype": "matmul", "input": [4], "output": [6],
         # "activation": lambda x: tf.nn.dropout(tf.nn.relu(x), tf.placeholder(tf.float32)),
         "activation": tf.nn.relu,
         "n_dim": 7 * 7 * 64,
         "m_dim": 1024
        },
        {"geneID": 6, 
         "nodetype": "matmul", "input": [5], "output": [7],
         "n_dim": 1024,
         "m_dim": 10
        },
        {"geneID": 7, 
         "nodetype": "noutput", "input": [6], "output": []
        }]
        
# McIntosh .... >_>
# init_genes = [{"geneID": 0, "nodetype": "ninput", "input": [], "output": [1]},
#         {"geneID": 1, "nodetype": "conv", "channels": [1, 8], "activation": tf.nn.relu6, "filtersize": 16, "input": [0], "output": [2]},
#         {"geneID": 2, "nodetype": "conv", "channels": [8, 16], "activation": tf.nn.relu6, "filtersize": 16, "input": [1], "output": [3]},
#         {"geneID": 3, "nodetype": "noutput", "activation": tf.nn.relu6, "input": [2], "output": []}]

basedir = "/Users/Bhavana/OneDrive/Documents/School/15-16/Winter/CNS 186/project/code/tf_logs"
nn = ann.NeuralNetwork(init_genes)
# runner = ann.NetworkRunner(basedir, x_train_both, y_train_both, x_test1, y_test1)
print(nn)
# stats1 = runner.train_network(nn, subdir="/m_reimpl", epochs=100, batch_size=10, learn_rate=0.0001)



