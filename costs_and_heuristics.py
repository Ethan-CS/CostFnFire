import random

import networkx as nx
from matplotlib import pyplot as plt

from choices import Heur, CFn


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


class Heuristic:
    heuristic_names = {Heur.RANDOM: "Random",
                       Heur.DEGREE: "Degree",
                       Heur.THREAT: "Threat",
                       Heur.COST: "Cost",
                       (Heur.DEGREE, Heur.THREAT): "Degree/Threat",
                       (Heur.DEGREE, Heur.COST): "Degree/Cost",
                       (Heur.THREAT, Heur.DEGREE): "Threat/Degree",
                       (Heur.THREAT, Heur.COST): "Threat/Cost",
                       (Heur.COST, Heur.DEGREE): "Cost/Degree",
                       (Heur.COST, Heur.THREAT): "Cost/Threat"}

    def __init__(self, which_heuristic: Heur, tie_break: Heur = None):
        if which_heuristic is None:
            raise ValueError("Invalid heuristic choice: {}".format(which_heuristic))
        if tie_break is not None:
            assert not tie_break == which_heuristic, ("can't break any ties with the same heuristic! "
                                                      f"heuristic: {which_heuristic}, tie break {tie_break}")
        if type(which_heuristic) is Heur:
            self.which_heuristic = which_heuristic
        else:
            raise TypeError("Invalid heuristic choice: {} of type {}".format(which_heuristic, type(which_heuristic)))

        self.which_heuristic = which_heuristic
        self.tie_break = tie_break

    def choose(self, graph, protected, burning, costs, threat_dict=None):
        if threat_dict is None:
            threat_dict = populate_threat_dict(graph, burning, protected)
        open_vertices = graph.nodes() - protected - burning
        degrees = dict(graph.degree())

        def heuristic_value(x, heuristic):
            if heuristic == Heur.DEGREE:
                return degrees[x]
            elif heuristic == Heur.THREAT:
                return threat_dict[x]
            elif heuristic == Heur.COST:
                return costs[x]
            else:
                return random.choice(list(open_vertices))

        if self.tie_break is None:
            key_func = lambda x: heuristic_value(x, self.which_heuristic)
        else:
            key_func = lambda x: (heuristic_value(x, self.which_heuristic),
                                  heuristic_value(x, self.tie_break))

        return min(open_vertices, key=key_func)


class CostFunction:
    def __eq__(self, __value):
        if type(__value) is CostFunction:
            return self.function == __value.function and self.cost == __value.cost
        else:
            return False

    def __str__(self):
        return str(self.dict_of_costs)

    def __init__(self, function: CFn, max_cost=-1):
        self.dict_of_costs = None
        self.function = function
        self.max_cost = max_cost

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
                    if self.max_cost < 0:
                        self.max_cost = 5  # default to something
                    self.dict_of_costs[vertex] = random.randint(1, self.max_cost)
                return self.dict_of_costs
            case CFn.UNIFORM:
                for vertex in graph.nodes():
                    self.dict_of_costs[vertex] = value
                return self.dict_of_costs
            case CFn.HESITANCY_BINARY:
                probability = 0.2972
                for vertex in graph.nodes():
                    r = random.random()
                    self.dict_of_costs[vertex] = 1 if r > probability else 0
                return self.dict_of_costs

    def stochastic_threat_cost(self, threat_dict, sf=10, value=1):
        for vertex in threat_dict:
            if self.max_cost > 0:
                self.dict_of_costs[vertex] = max(min(int(threat_dict[vertex]/sf) + random.randint(-value, value),
                                                     self.max_cost), 0)
            else:
                self.dict_of_costs[vertex] = max(threat_dict[vertex] + random.randint(-value, value), 0)
        return self.dict_of_costs


def test_heuristic():
    g = nx.erdos_renyi_graph(100, 0.3)
    burn = {0}
    protecc = set()
    print('list of degrees:', nx.degree(g))
    cost = CostFunction(CFn.UNIFORM)
    dict_costs = cost.cost(g, {})
    print('costs:', dict_costs)
    threat_dict = populate_threat_dict(g, burn, protecc)
    print('threats:', threat_dict)
    print("-----")
    print('DEGREE:', Heuristic(Heur.DEGREE).choose(g, protecc, burn, dict_costs, threat_dict))
    print('DEGREE+THREAT:',
          Heuristic(Heur.DEGREE, Heur.THREAT).choose(g, protecc, burn, dict_costs, threat_dict))
    print('DEGREE+COST:',
          Heuristic(Heur.DEGREE, Heur.COST).choose(g, protecc, burn, dict_costs, threat_dict))
    print("-----")
    print('THREAT:', Heuristic(Heur.THREAT).choose(g, protecc, burn, dict_costs, threat_dict))
    print('THREAT+DEGREE:',
          Heuristic(Heur.THREAT, Heur.DEGREE).choose(g, protecc, burn, dict_costs, threat_dict))
    print('THREAT+COST:',
          Heuristic(Heur.THREAT, Heur.COST).choose(g, protecc, burn, dict_costs, threat_dict))
    print("-----")
    print('COST:', Heuristic(Heur.COST).choose(g, protecc, burn, dict_costs, threat_dict))
    print('COST+DEGREE',
          Heuristic(Heur.COST, Heur.DEGREE).choose(g, protecc, burn, dict_costs, threat_dict))
    print('COST+THREAT:',
          Heuristic(Heur.COST, Heur.THREAT).choose(g, protecc, burn, dict_costs, threat_dict))


# test_heuristic()


def test_cost_function():
    for c in [CFn.STOCHASTIC_THREAT_LO, CFn.STOCHASTIC_THREAT_HI, CFn.UNIFORMLY_RANDOM, CFn.HESITANCY_BINARY]:
        print(f' === {c} ===')
        cost_function = CostFunction(c, max_cost=10)
        g = nx.random_lobster(100, 0.2, 0.05)
        threats = populate_threat_dict(g, {0}, set())
        costs = cost_function.cost(g, threats)

        print('costs:', len(costs), costs)
        print('from threat dict:', threats)

        # Histogram
        plt.hist(list(costs.values()), label=str(c).split('.')[1].replace('_', ' '), bins=max(costs.values()), alpha=0.4, density=True)
        plt.legend()
        plt.show()


# test_cost_function()
