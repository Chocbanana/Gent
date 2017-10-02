import ujson as json
import os

import torch
from torch.autograd import Variable
import torch.nn as nn

def load_unpickleable(seq):
    if isinstance(seq[7]["activation"], (list, tuple)):
        seq[7]["activation"] = [eval(x) for x in seq[7]["activation"]] 
    else:
        seq[7]["activation"] = eval(seq[7]["activation"])
        
    if isinstance(seq[12]["in_activation"], (list, tuple)):
        seq[12]["in_activation"] = [eval(x) for x in seq[12]["in_activation"]]
    else:
        seq[12]["in_activation"] = eval(seq[12]["in_activation"])
    return seq

def from_json(model_name, geneseq, connections, inputs, outputs):
    if model_name.startswith("MT-"):
        geneseq = load_unpickleable(geneseq)
    
    # Json doesnt like numbers as keys >:(
    connections = dict(zip(map(int, connections.keys()), connections.values()))
    
    return geneseq, connections, inputs, outputs

def MT_model_def(nfactors, nhid1, nencode1, nencode2, n_types, n_maps, nhid2, npredict, 
                out_activations, in_act, tied=True, no_rnn=False, savepath=None):
    seq = [{"gid": 1, "ntype": "nn_in", "d_in": [-1, nfactors], "d_out": [-1, nfactors]},
           {"gid": 2, "ntype": "nn_in", "d_in": [-1, 1], "d_out": [-1, 1]},
           {"gid": 3, "ntype": "nn_in", "d_in": [-1, 1], "d_out": [-1, 1]},
           {"gid": 4, "ntype": "lin", "d_in": [-1, nfactors], "d_out": [-1, 1, nhid1]},
           {"gid": 5, "ntype": "embed", "d_in": [-1, 1], "d_out": [-1, 1, nencode1], "n_embed": n_types},
           {"gid": 6, "ntype": "embed", "d_in": [-1, 1], "d_out": [-1, 1, nencode2], "n_embed": n_maps},
           {"gid": 7, "ntype": "lstm", "d_in": [-1, 1, nhid1 + nencode1 + nencode2], "d_out": [-1, nhid2],
            "d_hidden": nhid2, "d_batch": 1, "num_layers": 1},
           {"gid": 8, "ntype": "lin", "d_in": [-1, nhid2], "d_out": [-1, npredict], 
            "activation": out_activations},
           {"gid": 9, "ntype": "lin", "d_in": [-1, nhid2], "d_out": [-1, nencode1]},
           {"gid": 10, "ntype": "lin", "d_in": [-1, nhid2], "d_out": [-1, nencode2]},
           {"gid": 11, "ntype": "lin", "d_in": [-1, nencode1], "d_out": [-1, n_types], "tied": 5 if tied else None},
           {"gid": 12, "ntype": "lin", "d_in": [-1, nencode2], "d_out": [-1, n_maps], "tied": 6 if tied else None},
           {"gid": 13, "ntype": "nn_out", "d_in": [-1, npredict + 2], "d_out": [-1, npredict + 2],
            "in_activation": in_act}
          ]
    
    if no_rnn:
        seq[6] = {"gid": 7, "ntype": "lin", "d_in": [-1, 1, nhid1 + nencode1 + nencode2], "d_out": [-1, nhid2]}
         
    connections = [([], [4]), ([], [5]), ([], [6]), ([1], [7]),
                   ([2], [7]), ([3], [7]), ([4, 5, 6], [8, 9, 10]), ([7], [13]),
                   ([7], [11]), ([7], [12]), ([9], [13]), ([10], [13]),
                   ([8, 11, 12], []), 
                  ]
    connections = dict(zip(range(1, len(seq) + 1), connections))
    
    inputs = (1, 2, 3)
    outputs = (8, 11, 12)
    
    if savepath is not None:
        if not os.path.exists(savepath):
                os.makedirs(savepath)
        with open(savepath+"model_def.json", "w") as file:
            file.write("[" + json.dumps(seq, indent=2)+',\n')
            file.write(json.dumps(connections)+',\n')
            file.write(json.dumps(inputs)+',\n')
            file.write(json.dumps(outputs)+']\n')
    
    return load_unpickleable(seq), connections, inputs, outputs


def test_model_def(nfactors=63, nencode=140, nhidden=320, npredict=64, rnn_type="gru", savepath=None):
    seq = [ { "gid": 0, "ntype": "nn_in", "d_in": [-1, nfactors], "d_out": [-1, nfactors] },
                { "gid": 1, "ntype": "lin", "d_in": [-1, nfactors], "d_out": [-1, 1, nencode] },
                { "gid": 2, "ntype": rnn_type, "d_in": [-1, 1, nencode], "d_out": [-1, nhidden]
                , "d_hidden": nhidden, "d_batch": 1, "num_layers": 2 },
                { "gid": 3, "ntype": "lin", "d_in": [-1, nhidden], "d_out": [-1, npredict] },
                { "gid": 4, "ntype": "nn_out", "d_in": [-1, npredict], "d_out": (-1, npredict) }
           ]

    connections = [([], [1]), ([0], [2]), ([1], [3]), ([2], [4]), ([3], []),]
    connections = dict(zip(range(0, len(seq)), connections))
    
    if savepath is not None:
        if not os.path.exists(savepath):
                os.makedirs(savepath)
        with open(savepath+"model_def.json", "w") as file:
            file.write("[" + json.dumps(seq, indent=2)+',\n')
            file.write(json.dumps(connections)+',\n')
            file.write(json.dumps([0])+',\n')
            file.write(json.dumps([4])+']\n')
    
    return seq, connections, [0], [4]