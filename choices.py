from enum import Enum


class Heur(Enum):
    RANDOM = 0,
    DEGREE = 1,
    THREAT = 2,
    COST = 3


dict_of_heur_names = {Heur.RANDOM: "Random",
                      Heur.DEGREE: "Degree",
                      Heur.THREAT: "Threat",
                      Heur.COST: "Cost",
                      }
to_add = dict()
for h1 in dict_of_heur_names:
    to_add[(h1, None)] = f"{dict_of_heur_names[h1]}"  # i.e. no tie breaks
    for h2 in dict_of_heur_names:
        if h1 != h2 and (h1, h2) not in to_add:
            to_add[(h1, h2)] = f"{dict_of_heur_names[h1]}/{dict_of_heur_names[h2]}"
dict_of_heur_names.update(to_add)

class CFn(Enum):
    UNIFORM = 0,  # same val for all
    UNIFORMLY_RANDOM = 1,  # each cost rand int from 1 to defined upper bound
    STOCHASTIC_THREAT = 2,
    STOCHASTIC_THREAT_LO = 3,  # threat with some random noise - lo/med/hi
    STOCHASTIC_THREAT_MID = 4,
    STOCHASTIC_THREAT_HI = 5,
    HESITANCY_BINARY = 6,  # 0 with probability = hesitancy rate, 1 otherwise


dict_of_cost_names = {CFn.UNIFORM: 'Uniform',
                      CFn.UNIFORMLY_RANDOM: 'Uniformly Random',
                      CFn.STOCHASTIC_THREAT: 'Threat with Stochasticity',
                      CFn.STOCHASTIC_THREAT_LO: 'Threat with Stochasticity (low)',
                      CFn.STOCHASTIC_THREAT_MID: 'Threat with Stochasticity (mid)',
                      CFn.STOCHASTIC_THREAT_HI: 'Threat with Stochasticity (high)',
                      CFn.HESITANCY_BINARY: 'Hesitancy - Binary Distribution'
                      }
