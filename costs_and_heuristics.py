import random
from enum import Enum

import networkx as nx
from matplotlib import pyplot as plt


def populate_threat_dict(graph: nx.Graph, burning: set, protected: set):
    open_vertices = graph.nodes() - protected - burning
    threats = {}
    for vertex in open_vertices:
        current_shortest = len(open_vertices)
        for fire in burning:
            dist = nx.shortest_path_length(graph, fire, vertex) if nx.has_path(graph, fire, vertex) else 0
            if dist < current_shortest:
                current_shortest = dist
        threats[vertex] = current_shortest
    return threats


class HeuristicChoices(Enum):
    RANDOM = 0,
    DEGREE = 1,
    THREAT = 2,
    COST = 3

    def __str__(self):
        names = {HeuristicChoices.RANDOM: "Random",
                 HeuristicChoices.DEGREE: "Degree",
                 HeuristicChoices.THREAT: "Threat",
                 HeuristicChoices.COST: "Cost"}
        return names[self]

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_string(s: str):
        names = {"random": HeuristicChoices.RANDOM,
                 "degree": HeuristicChoices.DEGREE,
                 "threat": HeuristicChoices.THREAT,
                 "cost": HeuristicChoices.COST}
        return names[s.lower()]

    @classmethod
    def index_of(cls, which_heuristic):
        return list(cls).index(which_heuristic)


class Heuristic:
    def __init__(self, which_heuristic: HeuristicChoices, tie_break: HeuristicChoices = None):
        if which_heuristic is None:
            raise ValueError("Invalid heuristic choice: {}".format(which_heuristic))
        if tie_break is not None:
            assert not tie_break == which_heuristic, ("can't break any ties with the same heuristic! "
                                                      f"heuristic: {which_heuristic}, tie break {tie_break}")
        if isinstance(which_heuristic, HeuristicChoices):
            self.which_heuristic = which_heuristic
        else:
            raise TypeError("Invalid heuristic choice: {} of type {}".format(which_heuristic, type(which_heuristic)))

        self.which_heuristic = which_heuristic
        self.tie_break = tie_break

    def __str__(self):
        heuristic_names = {HeuristicChoices.RANDOM: "Random",
                           HeuristicChoices.DEGREE: "Degree",
                           HeuristicChoices.THREAT: "Threat",
                           HeuristicChoices.COST: "Cost"}
        if self.tie_break is None:
            if self.which_heuristic is None:
                return ""
            return heuristic_names[self.which_heuristic]
        return f'{heuristic_names[self.which_heuristic]}/{heuristic_names[self.tie_break]}'

    def __repr__(self):
        return self.__str__()

    def choose(self, graph, protected, burning, costs, threat_dict=None):
        if threat_dict is None:
            threat_dict = populate_threat_dict(graph, burning, protected)
        open_vertices = graph.nodes() - protected - burning
        degrees = dict(graph.degree())

        def heuristic_value(x, heuristic):
            if heuristic == HeuristicChoices.DEGREE:
                return degrees[x]
            elif heuristic == HeuristicChoices.THREAT:
                return threat_dict[x]
            elif heuristic == HeuristicChoices.COST:
                return costs[x]
            else:
                return random.choice(list(open_vertices))

        if self.tie_break is None:
            key_func = lambda x: heuristic_value(x, self.which_heuristic)
        else:
            key_func = lambda x: (heuristic_value(x, self.which_heuristic),
                                  heuristic_value(x, self.tie_break))

        return min(open_vertices, key=key_func)

    @classmethod
    def from_string(cls, h: str):
        if '/' in h:
            h1, h2 = h.split('/')
            return cls(HeuristicChoices.from_string(h1), HeuristicChoices.from_string(h2))
        else:
            return cls(HeuristicChoices.from_string(h))


class CFn(Enum):
    UNIFORM = 0,  # same val for all
    UNIFORMLY_RANDOM = 1,  # each cost rand int from 1 to defined upper bound
    STOCHASTIC_THREAT = 2,
    STOCHASTIC_THREAT_LO = 3,  # threat with some random noise - lo/med/hi
    STOCHASTIC_THREAT_MID = 4,
    STOCHASTIC_THREAT_HI = 5,
    HESITANCY_BINARY = 6,  # 0 with probability = hesitancy rate, 1 otherwise

    def __str__(self):
        names = {CFn.UNIFORM: "Uniform",
                 CFn.UNIFORMLY_RANDOM: "Uniformly Random",
                 CFn.STOCHASTIC_THREAT: "Stochastic Threat",
                 CFn.STOCHASTIC_THREAT_LO: "Stochastic-Threat (low)",
                 CFn.STOCHASTIC_THREAT_MID: "Stochastic-Threat (mid)",
                 CFn.STOCHASTIC_THREAT_HI: "Stochastic-Threat (high)",
                 CFn.HESITANCY_BINARY: "Hesitancy-Binary"}
        return names[self]

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_string(cls, function):
        names = {"uniform": CFn.UNIFORM,
                 "uniformly random": CFn.UNIFORMLY_RANDOM,
                 "stochastic threat": CFn.STOCHASTIC_THREAT,
                 "stochastic threat low": CFn.STOCHASTIC_THREAT_LO,
                 "stochastic threat mid": CFn.STOCHASTIC_THREAT_MID,
                 "stochastic threat high": CFn.STOCHASTIC_THREAT_HI,
                 "hesitancy binary": CFn.HESITANCY_BINARY}
        return names[function.lower()]


class CostFunction:
    def __init__(self, function, max_cost=2, min_cost=1):
        self.dict_of_costs = None
        self.max_cost = max_cost
        self.min_cost = min_cost

        if isinstance(function, CFn):
            self.function = function
        elif isinstance(function, str):
            self.function = CFn.from_string(function)
        else:
            raise ValueError(f"Invalid cost function: {function} of type {type(function)}")

    # def __eq__(self, __value):
    #     if type(__value) is CostFunction:
    #         return self.function == __value.function and self.cost == __value.cost
    #     else:
    #         return False

    def __str__(self):
        dict_of_cost_names = {CFn.UNIFORM: 'Uniform',
                              CFn.UNIFORMLY_RANDOM: 'Uniformly Random',
                              CFn.STOCHASTIC_THREAT: 'Threat with Stochasticity',
                              CFn.STOCHASTIC_THREAT_LO: 'Threat with Stochasticity (low)',
                              CFn.STOCHASTIC_THREAT_MID: 'Threat with Stochasticity (mid)',
                              CFn.STOCHASTIC_THREAT_HI: 'Threat with Stochasticity (high)',
                              CFn.HESITANCY_BINARY: 'Hesitancy - Binary Distribution'
                              }
        return f'function: {dict_of_cost_names[self.function]} \ncosts:,  {self.dict_of_costs}'

    def __repr__(self):
        return self.__str__()

    # TODO discuss w Jess how values and distributions are defined,
    #  should make them more directly comparable
    def cost(self, graph, threat_dict, value=1):
        self.dict_of_costs = {}
        if self.function == CFn.STOCHASTIC_THREAT_LO:
            value = 1
            self.function = CFn.STOCHASTIC_THREAT
        elif self.function == CFn.STOCHASTIC_THREAT_MID:
            value = 2
            self.function = CFn.STOCHASTIC_THREAT
        elif self.function == CFn.STOCHASTIC_THREAT_HI:
            value = 3
            self.function = CFn.STOCHASTIC_THREAT

        match self.function:
            case CFn.STOCHASTIC_THREAT:
                return self.stochastic_threat_cost(threat_dict, value)
            case CFn.UNIFORMLY_RANDOM:
                for vertex in graph.nodes():
                    self.dict_of_costs[vertex] = random.randint(self.min_cost, self.max_cost)
                return self.dict_of_costs
            case CFn.UNIFORM:
                for vertex in graph.nodes():
                    self.dict_of_costs[vertex] = max(value, self.min_cost)
                return self.dict_of_costs
            case CFn.HESITANCY_BINARY:
                probability = 0.2972
                for vertex in graph.nodes():
                    r = random.random()
                    self.dict_of_costs[vertex] = self.max_cost if r > probability else self.min_cost
                return self.dict_of_costs

    def stochastic_threat_cost(self, threat_dict, sf=10, value=1):
        for vertex in threat_dict:
            self.dict_of_costs[vertex] = max(min(int(threat_dict[vertex] / sf) + random.randint(-value, value), self.max_cost), self.min_cost)
        return self.dict_of_costs
