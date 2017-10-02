# Gent
A library for the easy construction, training, and running of artifical neural networks; built on top of pytorch.

Eventually, this library will be "An algorithm for Genetic Evolution of Network Topologies"
 
## Using Gent

Only 4 steps needed to make and run a neural network:
```python
import Gent as ann

gen = ann.NNGenome(geneseq)
network = ann.NeuralNetwork(gen.allGenes, connections, inputs, outputs) 
runner = ann.NetworkRunner(network, xtr, ytr)
runner.train()
```
Where the arguments are:
- geneseq: A list of dicts/sequences to define the layers of a network where the required/allowed keys are defined in `./NNGenome`.
- connections ({int: ([int], [int])}): Dict with `key= gid, value= tuple of lists ([in], [out])` of gids, for each corresponding gene in the genes list, giving the input and output for that gid key
- inputs (optional): Which layers the data is expected to go to.
- outputs (optional): List of output. Loss during training is applied to these.
- xtr, ytr: Data that is of the type torch.Tensor; can be lists of Tensors or a single Tensor. 

See `./NetworkRunner` for potential args to give to the runner.

**Files saved by Gent:**

- **~/model**: The torch-serialized weights for the network created by calling network.state_dict(), which outputs all saved gradients and values in registered Parameters.
- **~/stats**: Test-level data; a pickled dict where {"loss"= list of loss per epoch, "err" = list of error per epoch}
- **~/stats_per_epoch**: Train-level data; a pickled list, length of number of epochs, where each value is a dict of {"loss"= list of loss per batch in epoch, "avg_err"/"perplexity" = list of that type of err per batch in epoch}

**Other files saved in the directory:**
- **~/model_def.json**: Saved by ./examples/ModelDefs.py, a json serialized list of 4 values: geneseq, connections, inputs, outputs. To properly load call `ModelDefs.from_json(model_name, *json.loads(file.read()))`.
- **~/test_guids.json**: List of the guids of the matches used in the most recent test data.


## Pytorch Version
is 0.2.0, more specifically 0.2.0_1
