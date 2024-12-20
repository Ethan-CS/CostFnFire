import random

import numpy as np

from choices import CFn


def threat_cost(graph, value=1, threat_dict=None):
    if threat_dict is None:
        threat_dict = {}
    dict_of_costs = {}
    for vertex in threat_dict:
        dict_of_costs[vertex] = threat_dict[vertex]
    return dict_of_costs


def stochastic_threat_cost(threat_dict, value=1):
    dict_of_costs = {}
    for vertex in threat_dict:
        dict_of_costs[vertex] = threat_dict[vertex] + random.randint(-value, value)
        if dict_of_costs[vertex] < 0:
            dict_of_costs[vertex] = 1
    # print(dict_of_costs)
    return dict_of_costs


class CostFunction:
    def __init__(self, function: CFn):
        self.function = function

    def cost(self, graph, value=1, threat_dict=None):
        if threat_dict is None:
            threat_dict = {}
        dict_of_costs = {}

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
                return stochastic_threat_cost(threat_dict, value)
            case CFn.UNIFORMLY_RANDOM:
                dict_of_costs = {}
                max_val = 5
                for vertex in graph.nodes():
                    dict_of_costs[vertex] = random.randint(1, max_val)
                return dict_of_costs
            case CFn.UNIFORM:
                for vertex in graph.nodes():
                    dict_of_costs[vertex] = value
                return dict_of_costs
            case CFn.HESITANCY_NORMAL:  # NB Designed for HIGH budgets e.g. defending one vertex should cost on avg. 25
                mean = 29.72
                std_dev = 10
                for vertex in graph.nodes():
                    init_cost = random.normalvariate(mean, std_dev)  # Generate a normally distributed cost
                    dict_of_costs[vertex] = int(np.clip(round(init_cost), 1, 100))
                return dict_of_costs
            case CFn.HESITANCY_BINARY:
                probability = 0.2972
                for vertex in graph.nodes():
                    r = random.random()
                    dict_of_costs[vertex] = 1 if r > probability else 0
                return dict_of_costs
