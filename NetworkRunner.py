#! python3.6
"""
Trains and runs an artificial neural network created by `NeuralNetwork`,
using pytorch.

Author: Bhavana Jonnalagadda, 2017
"""
import os
import math
import pickle
import random
import numpy as np

from NeuralNetwork import NeuralNetwork, _var, _tensor
# import NNGenome as gn


import torch, torch.optim
import torch.autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


class NetworkRunner:
    """Runs a neural network. 

    Attributes:
        network (`NeuralNetwork`): The actual initialized network to run.
        seq: Whether the given data should be treated as sequential.
        dropout: Whether to do dropout (not recommended).
    """
    errtypes = {
                "avg_err": (lambda x: torch.mean(x)),
                "perplexity": (lambda x: math.exp(x))
               }

    def __init__(self, network , xtr, ytr, xte=None, yte=None, seq=True, dropout=False):
        self.network = network
        self.dropout = dropout
        self.seq = seq

        if xte is not None:
            self.x_train = xtr
            self.y_train = ytr
            self.x_test = xte
            self.y_test = yte
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = self._make_test_data(xtr, ytr)

    def evaluate(self, data, data_y=None, criterion=None, batched=None, err=None):
        """Runs the network WITHOUT training it.

        Args:
            data: Given x data (which has not been turned into a `torch.autograd.Variable` yet)
            data_y (optional): If given, the y data to compare against using the rest
                of the args.
            criterion: The loss function(s).
            batched: If given, the batch size to split the data into (since apparently pytorch
                with Cuda can't handle some too big of datasets stored in GPU memory)
            err (optional): If given, the type of error to calculate.

        Returns:
            result (list): A list of all the results which varies in length based on what was
                requested -- (y_pred, loss (optional), error (optional))
        """
        # Take out of training mode, into eval mode
        if self.dropout:
            self.network.eval()

        # Clear the gradients
        self.network.zero_grad()

        # Enable GPU optimization
        if torch.cuda.is_available():
            self.network.cuda()

        # Whether there are multiple inputs/outputs
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

                # Run the network
                if mult_out:
                    multi = self.network(x_)
                    for j in range(mult_out): results[j].append(multi[j])
                else:
                    results.append(self.network(x_))

                print("{}".format(i), end=' ') 

            # Concatenate the batched output back together
            y_pred = [torch.cat(r, dim=0) for r in results] if mult_out else torch.cat(results, dim=0)
            print()

        # Not batching input
        else:
            # Run the network
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
                      lossfcn="MSELoss", opt="SGD", err="avg_err", clipgrad=0.25, interval=50, l2=0.00, path=None, pretrained=None):
        """Trains the network on the data using the following given parameters.

        Args:
            epochs: The number of times to train the network over all the data.
            batch_size: The number of samples to include in each batch fed to the network.
            learn_rate: The rate by which the network weights are adjusted by the 
                optimizer (smaller numbers are more stable but take longer for training) 
            lossfcn ([str], str): Loss function(s) applied to the output. If a list, length
                must be the same as number of outputs.
            opt: The optimizer used.
            err: The type of error calculated, must be a key in `NetworkRunner.errtypes`.
            clipgrad: The maximum norm of the gradients when they are clipped 
                (to prevent exploding gradients).
            interval: The interval (over all batches) at which to report and save checkpoints.
            l2: If non-zero, scales the L2 regularization applied by the optimizer (smaller 
                means less regularization, 0.0 is no regularization).
            path: Where to save the data/checkpoints.
            pretrained: If given, a dict of the pretrained data from a previous training 
                that has {"model": self.network.state_dict(), "data": data, 
                "stats_per_epoch": stats_per_epoch, "starting_epoch": epochs} i.e. what is
                saved at the checkpoints and end of training.
        """
        # Load the weights if given
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
                # Create batches
                if self.seq:
                    batches = [slice(i, i+bsz) for i in range(0, len_data - 1, bsz) \
                                if i + bsz < len_data]
                    if batches[-1].stop < len_data:
                        batches.append(slice(batches[-1].stop, len_data))
                # (of different random data points each epoch if not a sequence)
                else:
                    inds = np.random.permutation(range(len_data))
                    batches = [torch.LongTensor(inds[i:(i + bsz)]) for i in range(0, len(inds), bsz) \
                                if i + bsz < len(inds)]

                stats = {}
                stats["loss"] = [0]
                stats[err] = []
                # Loop over batched data
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
                    # Loss for single output
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

                    # Reduce exploding gradient problem
                    if clipgrad:
                        nn.utils.clip_grad_norm(self.network.parameters(), clipgrad)

                    # Update weights of the network
                    optimizer.step()

                    # Report and save performance for batch interval
                    if b % interval == 0 and b != 0:
                        # Save stats
                        avgloss = stats["loss"][-1] / interval
                        stats["loss"][-1] = avgloss
                        stats[err].append(self._calc_err(err, loss=avgloss, y_pred=y_pred, y_=y_, len=len(loss)))
                        stats["loss"].append(0)

                        print("Epoch: {}\t Batch: {}/{}\t Loss: {}\t {}: {}\t".format(
                                e, b, len(batches), avgloss, err, stats[err][-1]))

                        

                # Run on test data
                y_pred, loss, error = self.evaluate(self.x_test, self.y_test, criterion, batched=500, err=err)

                # Save stats for epoch
                stats_per_epoch.append(stats)
                data["loss"].append(loss)
                data["err"].append(error)

                # Save to file
                if path: self._save_data(path+"checkpoints/epoch_{}-".format(e), stats=data, stats_per_epoch=stats_per_epoch, model=self.network)

                # Report on test data
                print("TEST DATA -- \t Epoch: {}\t Loss: {}\t {}: {}\t".format(e, loss, err, error))


            # Save all to file after all epochs
            print("Finished {} epochs, saving to {}".format(epochs, path))
            if path: self._save_data(path, model=self.network, stats=data, stats_per_epoch=stats_per_epoch)

        # Save the data if user exits with CTRL-c
        except KeyboardInterrupt:
            print("Quitting from interrupt")

            stats["loss"][-1] = stats["loss"][-1] / (interval if b % interval == 0 else b % interval)
            stats[err].append(self._calc_err(err, loss=stats["loss"][-1], y_pred=y_pred, y_=y_, 
                                len=len(self.network.output_nodes)))
            stats_per_epoch.append(stats)
            
            print("Saving to {}".format(path))
            if path: self._save_data(path+"interrupt/", model=self.network, stats=data, stats_per_epoch=stats_per_epoch)
            print("Finished saving")


    """ Helper functions """

    def _make_test_data(self, data_x, data_y, amount=10):
        """
        Makes test data out of the given data if needed; Assumes the given 
        data was already pre-shuffled.

        Args:
            amount: the amount to divide by (the percent of data that will be test)
        """
        num = len(data_x) // amount
        return data_x[num:], data_y[num:], data_x[:num], data_y[:num]

        
    def _calc_err(self, errtype, **kwargs):
        """Calculate the error (unreasonably complicated for some reason).

        Args:
            errtype: Must be a key in `NetworkRunner.errtypes`.
            loss: The calculated loss.
            len: Number of outputs.
            y_pred: Output(s) from the network.
            y_: Given y data(s); must already be a torch Variable.
        """
        # Calculate perplexity
        if errtype == "perplexity":
            result = math.exp(kwargs["loss"])

        # Find average difference between predicted and actual y
        elif errtype == "avg_err":
            # If multiple outputs (such a bitch)
            if "len" in kwargs and kwargs["len"] and kwargs["len"] > 1:
                # Use stored activation functions to format the y_pred if needed
                if self.network.format_fcn:
                    y_pred = [self.network.format_fcn[i](kwargs["y_pred"][i]) for i in range(kwargs["len"])]
                else: 
                    y_pred = kwargs["y_pred"]

                # Sooo many bugs from this single line....
                # NOTE: tensor cannot be a LongTensor when doing subtraction :/
                # TODO: Do mean of absolute value of sub
                errs = [torch.mean(y_pred[i].sub_(kwargs["y_"][i]).type(_tensor("FloatTensor"))).data[0] for i in range(kwargs["len"])]
                result = sum(errs) / kwargs["len"]

            # Single output
            else:
                result = torch.mean((kwargs["y_pred"] - kwargs["y_"]).type(_tensor("FloatTensor"))).data[0]

        return result


    def _save_data(self, path, **kwargs):
        """Save the given data to file.
        """
        for key, value in kwargs.items():
            # Make the path if it doesn't already exist
            if not os.path.exists(path[0:path.rindex("/")]):
                os.makedirs(path[0:path.rindex("/")])

            with open(path + key, 'wb') as file:
                # Save the network trained weights
                if key == "model":
                    # Save as cpu tensors so the model can actually be used
                    # by computers without fucking cuda
                    value.cpu()
                    torch.save(value.state_dict(), file)
                    if torch.cuda.is_available(): value.cuda()

                else:
                    pickle.dump(value, file)
                print("Saved {}".format(key))

                