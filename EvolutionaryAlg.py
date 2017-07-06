"""
Author: Bhavana Jonnalagadda, 2016
"""

import random
import math

"""
OUTLINE FOR 3RD TERM:
NeuralNetwork:
    - NetworkRunner:
        - make flexible for assigning activation, regularization, optimizer, metrics all in general
        - Hyperparamter training!!
    - RecurrentNode:
        - Get working!

    - NeuralNetwork:
        - Use Graph library to store genes/representation
        - derive from Phenotype
    - NetworkNode:
    - Define NetworkGenome:
        - general rules for ALL networks (or distinguish b/w rnn and conv)

    - do remainings todos
    - save networks/graphs to file


EvolutionaryAlg:
    - Genome:
    - Phenotype:
        - compatibility
    - Population:
        - Fitness assessment
        - combine 
        - mutate
    - GeneticAlg:
        - make multithread-able
        - add debug printing
        - save histories of generations
            - print to file?

    - add asserts for checking proper types/structures/presence
    - force the virtual functions to be derive implemented
        - just, raise exception?


RESEARCH AFTER ABOVE IS COMPLETED:
- improve GA
- reimplement and evolve popular networks, use popular datasets
- activation functions in form of a*sin(bx + c) + d
- fractal patterns in topology, rnn output
- analyze emergent properties

--------------------------------------------

workflow:
    genome = Genome([rules])
    pop = Population(genome.makenew([initial_nn]), Network)
    env = NetworkRunner([params])

    ga = GeneticAlgorithm(genome, pop, env)
    next(ga)

TODO: consider which classes need to implement
    - __eq__
    - __ne__
    - __next__ / __iter__
    - __str__

"""


class Gene:
    pass

# GRAPH of genes
class Genotype:
    pass

# TODO: set containing all created genotypes
# TODO: constraints on param variation? 
#       - enforce param ranges and sets
class Genome:

    def __init__(self, rules):
        self._process_rules(rules)
        self.madeGenotypes = []
        self.madeGenes = {}

    

    def makenew(self, genes):
        pass

    # TODO: in population
    # def mate(self, parent1, parent2):
        
        # geneseq1 = set(parent1.geneseq.keys())
        # geneseq2 = set(parent2.geneseq.keys())

        # # find the genes that they have the same
        # same = geneseq1.intersection(geneseq2)
        # # randomly select the node for those genes from each
        # seq1 = random.sample(same, int(len(same)/2))
        # seq2 = same.difference(seq1)
        # # select the nodes from more fit parent
        # excess = geneseq1.difference(geneseq2)

        # mutate the new geneseq

        # construct the entity from the geneseq using Network


    def mutate(self, genotype):
        pass

    def _process_rules(self, rules):
        pass


# NOTE: NEAT modification, instead of excess/disjoint genes the diff ways to be different are weighted
class Phenotype:
    
    # derived class
    def __init__(self, genotype):
        pass

    # TODO
    # Implemented in derived class
    # return a number in [0, 1]
    def compatibility(self, m1, m2):
        pass

    # def are_compatible(self, network1, network2):
    #     # TODO: REDO
    #     geneseq1 = [n.gene for n in network1.nodes]
    #     geneseq2 = [n.gene for n in network2.nodes]

    #     ID1 = set([g["geneID"] for g in geneseq1])
    #     ID2 = set([g["geneID"] for g in geneseq2])

    #     # samegenes = ID1.intersection(ID2)
    #     # weightdiffs = []
    #     # for g in samegenes:
    #     #     node1 = next(n for n in network1.nodes if n.gene["geneID"] == g)
    #     #     if hasattr(node1, "weights"):
    #     #         node2 = next(n for n in network2.nodes if n["geneID"] == g)

    #     diffgenes1 = ID1.difference(ID2)
    #     diffgenes2 = ID2.difference(ID1)

    #     # diff_connections_count = 0
    #     # diff_params_count = 0
    #     # for g in diffgenes1:
    #     #     node1 = next(n for n in geneseq1 if n["geneID"] == g)

    #     # Compatibility threshold
    #     cD = 0.5
    #     # different genes
    #     c1 = 1
    #     # # different params
    #     # c2 = 1

    #     N = max([len(ID1), len(ID2)])

    #     # Compatibility distance formula
    #     delta = c1 * (len(diffgenes1) + len(diffgenes2)) / N

    #     return True if delta <= cD else False



# Population: all networks in pop, networks in each species
# Population:
#     - Speciation
#     - Compatability function: Use NEAT compatablity distance delta

class Population:

    def __init__(self, genotype, Phenotype):
        # TODO: where is seed made?
        # Seed = object/organism/network, not the gene dict
        seed = Phenotype(genotype)
        species = {"ID": 0, "rep": seed, "members": [seed], "fitnesses": [], "species_fitness": 0}
        self.sp_counter = 1
        self.species_list = [species]
        self.max_pop = 20
        self.members = {}


    def assignMember(self, id, fitness):
        pass

    # Re-sort into species
    def speciate(self):
        # get all members from across species
        all_members = [(m, f) for s in self.species_list for m, f in zip(s["members"], s["fitnesses"])]

        for s in self.species_list:
            s["species_fitness"] = 0
            s["members"] = []
            s["fitnesses"] = []

        # place all members in species and add their fitness to that species
        for m, f in all_members:
            # Find first species that member is compatible with
            species = next((s for s in self.species_list if self.genome.are_compatible(m, s["rep"])), None)
            # otherwise make a new species
            if species is None:
                self.species_list.append({"ID": self.sp_counter, "rep": m, "members":[m], "fitnesses": [f], "species_fitness": 0})
                self.sp_counter += 1
            else:
                species["members"].append(m)
                species["fitnesses"].append(f)


        # re-choose representative and calculate shared fitness:
        # adj_fitness(member) = fitness / (number of other members in same species)
        # shared_fitness(species) = sum(all adj_fitnesses) / (n*(n-1))
        for s in self.species_list:
            s["rep"] = random.choice(s["members"])
            s["species_fitness"] = sum(s["fitnesses"]) / (len(s["members"]) * (len(s["members"]) - 1))

        # TODO: delete empty species or species with no asisgned children


    # Make the new population
    # TODO: assert all members have been assigned a species and fitness
    # TODO:
        # - Move species rep calculation 
        # - get COMPATABILITY() for mating pairs
        # - COMBINE in a way that preserves every fit member's genetic info
        # - MUTATE as needed (to fill max pop)
    def repopulate(self, genome):
        total = sum([s["species_fitness"] for s in self.species_list])

        for s in self.species_list:
            # get parents
            max_children = int(Population.max_pop * s["species_fitness"] / total)
            top_members = sorted(zip(s["fitnesses"], range(len(s["members"]))))

            if len(s["members"]) == 1:
                parents = [s["members"][0], s["members"][0]]
            elif len(s["members"]) < 3:
                parents = s["members"]
            # get top 60%
            else:
                parents = [s["members"][m[1]] for m in top_members[0:int(len(top_members) * 0.06 + 0.5)]]

            parent_pairs = list(zip(parents[:-1], parents[1:]))

            # create the children
            children = []
            for i in range(max_children):
                child = self.genome.mate(*parent_pairs[i % len(parent_pairs)])
                children.append(child)

            # replace old population with new
            s["members"] = children
            s["fitnesses"] = []

class Environment:
    pass


# genome = Genome([rules])
# pop = Population(genome.makenew([initial_nn]), Network)
# env = NetworkRunner([params])

# ga = GeneticAlgorithm(genome, pop, env)
# next(ga)
class GeneticAlgorithm:


    def __init__(self, genome, population, environment):
        self.gen = genome
        self.pop = population
        self.env = environment

        self.generation = 0


    # Returns the next population generation
    def __next__(self):
        for i, p in self.pop.members.items():
            # assign fitness and put into a species
            self.pop.assignMember(i, self.env.eval(p, self.generation))

        # make new generation
        self.pop.repopulate(self.gen)
        self.generation += 1


    def __iter__(self):
        return self

    # def eval_population(self):
    #     for s in self.population.species_list:
    #         s["fitnesses"] = []
    #         for m in s["members"]:
    #             fitness = self.evaluator.eval(m, self.generation, s["ID"])
    #             s["fitnesses"].append(fitness)

