import random

import networkx as nx

from choices import Heur, CFn
from cost_function import CostFunction


def populate_threat_dict(graph, burning, protected):
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
    def __init__(self, which_heuristic, tie_break=None):
        if which_heuristic is None:
            raise ValueError("Invalid heuristic choice: {}".format(which_heuristic))
        if tie_break is not None:
            assert not tie_break == which_heuristic, ("can't break any ties with the same heuristic! "
                                                      f"heuristic: {which_heuristic}, tie break {tie_break}")
        if type(which_heuristic) is Heur:
            self.which_heuristic = which_heuristic
        elif type(which_heuristic) is list and len(which_heuristic) == 2:
            self.which_heuristic = which_heuristic[0]
            self.tie_break = which_heuristic[1]

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


dict_of_heuristic_names = {Heur.RANDOM: "Random",
                           Heur.DEGREE: "Degree",
                           Heur.THREAT: "Threat",
                           Heur.COST: "Cost",
                           (Heur.DEGREE, Heur.THREAT): "Degree/Threat",
                           (Heur.DEGREE, Heur.COST): "Degree/Cost",
                           (Heur.THREAT, Heur.COST): "Threat/Cost"}

g = nx.erdos_renyi_graph(100, 0.3)
burn = {0}
protecc = {}

print('list of degrees:', nx.degree(g))

cost = CostFunction(CFn.UNIFORM)
dict_costs = cost.cost(g)
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
