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



# TODO: docstrings for namedtuples in the form
# Gene.__doc__ = ": "
# Gene.field.__doc__ = " "


# class BaseGene(object):
#     __slots__ = ("gid", "ntype", "d_in", "d_out")
#
#     def __init__(self, **kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)
#
# class NNInputGene(BaseGene):
#     __slots__ = ()
#
#     def __init__(self, **kwargs):
#         super(NNInputGene, self).__init__(**kwargs)
#
# class NNOutputGene(BaseGene):
#     __slots__ = ("activation",)
#
#     def __init__(self, activation=None, **kwargs):
#         super(NNOutputGene, self).__init__(activation, **kwargs)


BaseGene = namedtuple("BaseGene", ["gid", "ntype", "d_in", "d_out"])

NNInputGene = namedtuple("NNInputGene", BaseGene._fields)

NNOutputGene = namedtuple("NNOutputGene", BaseGene._fields + ("activation",))
NNOutputGene.__new__.__defaults__ = (None,)

ConvGene = namedtuple("ConvGene", BaseGene._fields + ("d_filter", "activation", "stride", "dropout"))
ConvGene.__new__.__defaults__ = (None, [1, 1, 1, 1], False)

PoolGene = namedtuple("PoolGene", BaseGene._fields + ("d_kernel", "activation", "stride", "padding", "dropout"))
PoolGene.__new__.__defaults__ = (None, [1, 1, 1, 1], 0, False)

LinearGene = namedtuple("LinearGene", BaseGene._fields + ("activation", "dropout"))
LinearGene.__new__.__defaults__ = (None, False)

RnnGene = namedtuple("RnnGene", BaseGene._fields + ("d_hidden", "d_batch", "num_layers", "nonlin", "bidir"))
RnnGene.__new__.__defaults__ = (1, "tanh", False)

EmbedGene = namedtuple("EmbedGene", BaseGene._fields + ("d_embed", "dropout"))
EmbedGene.__new__.__defaults__ = (False,)

NodeTypes = {
            "nn_in": NNInputGene, "nn_out": NNOutputGene, "conv": ConvGene
            , "maxpool": PoolGene, "avgpool": PoolGene, "lin": LinearGene
            , "rnn": RnnGene, "lstm": RnnGene, "gru": RnnGene
            , "embed": EmbedGene
            }
GeneTypes = tuple(set(NodeTypes.values()))


allnodes = list(NodeTypes.keys())


# class NNGene(object):
#     def __init__(self, params):




# Will be a graph
# TODO: genotype will handle:
#     - checking that all have appropriate in/outputs
#         - if in/output not given, can infer from connected
#     - assigning unique nids (different from gid, which identifies.... something unique)
#         - TODO: still need to decide if gid applies jsut to ntype, or also params
#         - also including in/output is isomorphism, since is determined by connections and gid
class NNGenotype:

    def __init__(self, genes):
        pass

        


class Allow(Enum):
    noneOf = 0
    only = 1
    allOf = 2

# GeneRules = { "nn_in": {"input": (Allow.only, []), "output": (Allow.allOf)}
#             , "nn_out": {"output": (Allow.only, []), "input": (Allow.allOf)}
#             , "conv": }

# class NNGene:
#     # genespace = [("ntype", allnodes), (["input", "output"], NNGenome.GID[])]
#
#     def check_correctness(self, kwargs):
#         pass
#
#     def make_gene(self, args):
#         if (self.check_correctness(args)):
#             return args[1](args)

class NNGenome:

    def __init__(self, genes=None):
        # Indexed by gid
        self.allGenes = {}
        self.counter = 1

        if genes:
            if isinstance(genes, (list, tuple)):
                for g in genes: self.new_gene(g)
            else:
                self.new_gene(genes)

    # def make_gene(self, params):
    #     if params["ntype"] not in NodeTypes:
    #         raise ValueError("The given ntype {} is not allowed".format(params["ntype"]))
    #
    #     if "gid" not in params or params["gid"] in self.allGenes:
    #         params["gid"] = self._new_gid()
    #
    #     newgene = NodeTypes[params["ntype"]](**params)
    #     self.allGenes[gid] = newgene
    #
    #     return newgene

    def new_gene(self, gene):
        if gene["ntype"] not in NodeTypes:
            raise ValueError("The given ntype {} is not allowed".format(gene["ntype"]))

        if "gid" not in gene or gene["gid"] in self.allGenes:
            gene["gid"] = self._new_gid()

        if isinstance(gene, dict):
            self.allGenes[gene["gid"]] = NodeTypes[gene["ntype"]](**gene)
        elif isinstance(gene, (list, tuple)):
            self.allGenes[gene["gid"]] = NodeTypes[gene["ntype"]](*gene)

        return self.allGenes[gene["gid"]]


    def _new_gid(self):
        while self.counter in self.allGenes:
            self.counter += 1

        return self.counter

    #
    # {"d_in": (3, 4, 11), "activation": "relu"}
    # [([1], []), ([2], [0])]