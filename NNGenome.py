#! python3.6
"""
Defines the parameters used by each specific network layer/node type; the 
namedTuple class is used to enforce required paremeters. The genes are
used by `NeuralNetwork`. Fields (starting from the right) are optional
if namedtuple.__new__.__defaults__ are set for them.

Author: Bhavana Jonnalagadda, 2017
"""

from enum import Enum
from collections import namedtuple, OrderedDict

BaseGene = namedtuple("BaseGene", ["gid", "ntype", "d_in", "d_out"])

NNInputGene = namedtuple("NNInputGene", BaseGene._fields)

NNOutputGene = namedtuple("NNOutputGene", BaseGene._fields + ("activation", "in_activation"))
NNOutputGene.__new__.__defaults__ = (lambda x: x, ())

ConvGene = namedtuple("ConvGene", BaseGene._fields + ("d_filter", "activation", "stride", "dropout", "padding"))
ConvGene.__new__.__defaults__ = (lambda x: x, [1, 1, 1, 1], False, 0)

PoolGene = namedtuple("PoolGene", BaseGene._fields + ("d_kernel", "activation", "stride", "padding", "dropout"))
PoolGene.__new__.__defaults__ = (lambda x: x, [1, 1, 1, 1], 0, False)

LinearGene = namedtuple("LinearGene", BaseGene._fields + ("activation", "dropout", "tied"))
LinearGene.__new__.__defaults__ = (lambda x: x, False, None)

RnnGene = namedtuple("RnnGene", BaseGene._fields + ("d_hidden", "d_batch", "num_layers", "bidir"))
RnnGene.__new__.__defaults__ = (1, False)

EmbedGene = namedtuple("EmbedGene", BaseGene._fields + ("n_embed", "dropout", "max_norm"))
EmbedGene.__new__.__defaults__ = (False, None)

NodeTypes = {
            "nn_in": NNInputGene, "nn_out": NNOutputGene, "conv": ConvGene
            , "maxpool": PoolGene, "avgpool": PoolGene, "lin": LinearGene
            , "rnn": RnnGene, "lstm": RnnGene, "gru": RnnGene
            , "embed": EmbedGene
            }
GeneTypes = tuple(set(NodeTypes.values()))

class NNGenome:
    """ Takes a list of dicts/sequences, applies them to the appropriate gene and stores them. """

    def __init__(self, genes=None):
        # Indexed by gid
        self.allGenes = OrderedDict()
        self.counter = 1

        if genes:
            # Sequence of genes
            if isinstance(genes, (list, tuple)):
                for g in genes: self.new_gene(g)
            # Single gene
            else: self.new_gene(genes)

    def new_gene(self, gene):
        """Creates a gene from the given parameters. 
        The arg `gene` can be a dict or list; if a list, values must be in order.
        """
        if gene["ntype"] not in NodeTypes:
            raise ValueError("The given ntype {} is not allowed".format(gene["ntype"]))

        # Make gid if it doesn't have one
        if "gid" not in gene or gene["gid"] in self.allGenes:
            gene["gid"] = self._new_gid()

        if isinstance(gene, dict):
            self.allGenes[gene["gid"]] = NodeTypes[gene["ntype"]](**gene)
        elif isinstance(gene, (list, tuple)):
            self.allGenes[gene["gid"]] = NodeTypes[gene["ntype"]](*gene)

        return self.allGenes[gene["gid"]]

    def _new_gid(self):
        """Simple id creator.
        """
        while self.counter in self.allGenes:
            self.counter += 1
        return self.counter
