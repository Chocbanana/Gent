from enum import Enum
from collections import namedtuple

# - Rules need for genome:
# NO RULES FOR GENE. ONLY IN GENOME!!!!
#     - a defined context: 
#  (ntype, input??, graph (for genome level rules))
#     - a limit using the Allow enum on gene/subgraph properties
# 
# IN GENE:
#     - a defined space: either list, num range, python type, or a 
#       transformation into one of those
#       - DEFINED AS FUNCTIONS/LAMBDAS
#   
#     - a way to search/find the rule space???



BaseGene = namedtuple("BaseGene", ["gid", "ntype", "input", "output"])

NNInputGene = namedtuple("NNInputGene", BaseGene._fields)
NNOutputGene = namedtuple("NNOutputGene", BaseGene._fields + ("activation",))
NNOutputGene.__new__.__defaults__ = (None,)
ConvGene = namedtuple("ConvGene", BaseGene._fields + ("dim_filter", "activation", "stride"))
ConvGene.__new__.__defaults__ = (None, [1, 1, 1, 1])
PoolGene = namedtuple("PoolGene", BaseGene._fields + ("dim_kernel", "dim_in", "dim_out", "activation", "stride", "padding"))
PoolGene.__new__.__defaults__ = (None, [1, 1, 1, 1], 0)
LinearGene = namedtuple("LinearGene", BaseGene._fields + ("dim_in", "dim_out", "activation"))
LinearGene.__new__.__defaults__ = (None,)

NodeTypes = {"nn_in": NNInputGene, "nn_out": NNOutputGene, "conv": ConvGene
            , "recurrent": None, "maxpool": PoolGene, "avgpool": PoolGene
            , "lin": LinearGene}

allnodes = list(NodeTypes.keys())


class NNGenotype:

    def __init__(self, genes):

        


class Allow(Enum):
    noneOf = 0
    only = 1
    allOf = 2

# GeneRules = { "nn_in": {"input": (Allow.only, []), "output": (Allow.allOf)}
#             , "nn_out": {"output": (Allow.only, []), "input": (Allow.allOf)}
#             , "conv": }

class NNGene:
    # genespace = [("ntype", allnodes), (["input", "output"], NNGenome.GID[])]

    def check_correctness(kwargs):
        pass

    def make_gene(args):
        if (self.check_correctness(args)):
            return args[1](args)

class NNGenome:
    
