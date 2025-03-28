import networkx as nx
import matplotlib.pyplot as plt
import random


# code used for preliminary experiments in conference version of paper

def cost_firefighter(graph, burning, protected, choose, cost_function, num_rounds = 50, INFO = False, budget = 5):
  if INFO:
    print("\n\n\nNEW RUN")
  total_nodes = len(graph.nodes())

  last_burning = 0
  for _ in range(num_rounds):
    open = graph.nodes() - protected - burning
    if len(open) == 0:
      break
    threat_dict = calculate_threat(graph, protected, burning, cost_function)
    # print(threat_dict)
        # choose something to protect
    costs = cost_function(graph, protected, burning, threat_dict = threat_dict)
    if INFO:
      print("COST FUNCTION")
      print(costs)

    total_cost = 0
    while total_cost < budget:
      open = graph.nodes() - protected - burning
      if len(open) == 0:
         break
      node_to_protect = choose(graph, protected, burning, costs, threat_dict)
      total_cost += costs[node_to_protect]
      if total_cost < budget:
         protected.add(node_to_protect)

    neighbours = set()
    for v in burning:
      neighbours.update(set(graph.neighbors(v)))
    neighbours -= set(protected)
    burning.update(neighbours)

    if INFO:
      print("Burning nodes")
      print(burning)
      print("Protected nodes")
      print(protected)
    if (len(burning) + len(protected)) == total_nodes:
      break
    last_burning = len(burning)
  return len(graph.nodes()) - len(burning)


def uniform_cost(graph, protected, burning, value = 1, threat_dict = {}):
    dict_of_costs = {}
    for vertex in graph.nodes():
        dict_of_costs[vertex] = value
    return dict_of_costs


def uniform_random_cost(graph, protected, burning, value = 1, threat_dict = {}):
    dict_of_costs = {}
    max_val = 5
    for vertex in graph.nodes():
        dict_of_costs[vertex] = random.randint(1,max_val)
    return dict_of_costs

def threat_cost(graph, protected, burning, value = 1, threat_dict = {}):
    dict_of_costs = {}
    for vertex in threat_dict:
        dict_of_costs[vertex] = threat_dict[vertex]
    return dict_of_costs

def stochastic_threat_cost(graph, protected, burning, threat_dict, value = 1, spread=1):
    dict_of_costs = {}
    for vertex in threat_dict:
        dict_of_costs[vertex] = threat_dict[vertex] + random.randint(-spread, spread)
        if dict_of_costs[vertex] < 0:
    	      dict_of_costs[vertex] = 1
    # print(dict_of_costs)
    return dict_of_costs

def stochastic_threat_cost_low(graph, protected, burning, threat_dict, value = 1):
    return stochastic_threat_cost(graph, protected, burning, threat_dict, spread=1)

def stochastic_threat_cost_mid(graph, protected, burning,  threat_dict, value = 1):
    return stochastic_threat_cost(graph, protected, burning, threat_dict,  spread=2)

def stochastic_threat_cost_hi(graph, protected, burning, threat_dict, value = 1):
    return stochastic_threat_cost(graph, protected, burning, threat_dict, spread=3)

def random_choice(graph, protected, burning, costs, threat_dict = {}):
    open = graph.nodes() - protected - burning
    return random.choice(list(open))


def cheapest_choice(graph, protected, burning, costs, threat_dict = {}):
  open = graph.nodes() - protected - burning
  return min(open, key=lambda x: costs[x])

def calculate_threat(graph, protected, burning, cost_function):
  dict_threat = {}
  open = graph.nodes() - protected - burning
  for vertex in open:
    current_shortest = len(open)
    for fire in burning:
      dist = nx.shortest_path_length(graph, fire, vertex)
      if dist < current_shortest:
        current_shortest = dist
    dict_threat[vertex] = dist  # E: this should maybe be `current_shortest`
  return dict_threat

def threat_choice(graph, protected, burning, costs, threat_dict):
  open = graph.nodes() - protected - burning
  return min(open, key=lambda x: threat_dict[x])

def threat_cheapest(graph, protected, burning, costs, threat_dict):
  open = graph.nodes() - protected - burning
  threat_cheapest_dict = {}
  for v in threat_dict:
    if v in open:
      threat_cheapest_dict[v] = (threat_dict[v], costs[v])
  return min(open, key=lambda x: threat_cheapest_dict[x])

def cheapest_threat(graph, protected, burning, costs, threat_dict):
  open = graph.nodes() - protected - burning
  for v in threat_dict:
    if v in open:
      threat_cheapest[v] = (costs[v], threat_dict[v])
  return min(open, key=lambda x: threat_cheapest[x])

def degree_choice(graph, protected, burning, costs, threat_dict):
  open = graph.nodes() - protected - burning
  return min(open, key=lambda x: graph.degree(x))

def degree_cheapest(graph, protected, burning, costs, threat_dict):
  open = graph.nodes() - protected - burning
  degree_cheapest ={}
  # print ("OPEN IS "  + str(open))
  for v in costs:
    if v in open:
      degree_cheapest[v] = (graph.degree(v), costs[v])
  return min(degree_cheapest.keys(), key=lambda x: degree_cheapest[x])



# Example usage



# Now we want some basic experiments.
# start with: on graph type (start with ER), for each graph, cost function, heuristic how many are saved?


def run_er_expts(size, p, num_trials, cost_type = uniform_cost):
  trials = {}

  dict_of_choice_funs = {"degree_cheapest": degree_cheapest, "degree_choice": degree_choice, "threat_cheapest_cost": threat_cheapest,
                         "cost_only_choice": cheapest_choice}
                        #  , 'stochastic_threat_cost': stochastic_threat_cost}
  # dict_of_cost_funs = {'threat_cost': threat_cost}
  for _ in range(num_trials):
    # graph = nx.barabasi_albert_graph(size, p)
    graph = nx.random_regular_graph(p, size)
    # for cost_type in dict_of_cost_funs:
    for choice_type in dict_of_choice_funs:
        saved = cost_firefighter(graph, set([0]), set([]), dict_of_choice_funs[choice_type], cost_type, num_rounds=5)
        if ('ba', size, p,cost_type, choice_type) not in trials:
          trials[('ba', size, p,cost_type, choice_type)] = []
        trials[('ba', size, p,cost_type, choice_type)].append(saved)
  return trials


dict_of_cost_funs = {'uniform_cost': uniform_cost, 'uniform_random_cost': uniform_random_cost, 'stochastic_threat_cost_hi': stochastic_threat_cost_hi,
                     'threat_cost': threat_cost, 'stochastic_threat_cost_low': stochastic_threat_cost_low, 'stochastic_threat_cost_mid': stochastic_threat_cost_mid}
dict_of_cost_funs = { 'stochastic_threat_cost_low': stochastic_threat_cost_low, 'stochastic_threat_cost_mid': stochastic_threat_cost_mid,
                     'stochastic_threat_cost_hi': stochastic_threat_cost_hi, 'uniform_random_cost': uniform_random_cost, 'uniform_cost': uniform_cost,}

dict_of_cost_names={ 'stochastic_threat_cost_low': 'Low-stochasticity threat', 'stochastic_threat_cost_mid': 'Mid-stochasticity threat',
                     'stochastic_threat_cost_hi': 'High-stochasticity threat', 'uniform_random_cost': 'Uniform random', 'uniform_cost': 'Uniform Cost'}
dict_of_choice_names = {"degree_cheapest": 'Degree/cost', "degree_choice": 'Degree', "threat_cheapest_cost": 'Threat/cost',
                         "cost_only_choice": 'Cost'}

for cost_type in dict_of_cost_funs:
  plt.clf()
  results = run_er_expts(50, 6, 1000, cost_type=dict_of_cost_funs[cost_type])
  for guy in results:
    print(guy, results[guy])
    plt.hist(results[guy], alpha = 0.3, label = dict_of_choice_names[guy[4]], density=True)
    plt.xlim(0,40)
    plt.ylim(0,0.5)
  plt.ylabel('Frequency')
  plt.xlabel('Number of nodes saved')
  plt.title(dict_of_cost_names[cost_type])
  plt.legend()
  plt.show()


# next task, add a budget and multiple defences - done
# next task, generate interesting experiments
# Possible interesting experiments: geometric random graphs and on random regular graphs what
# heuristics do better along with which cost functions
# add a periodic cost function?
# a heuristic that is in-proportion stochastically to cost?
# OR even better a cost that is stochastically related to threat
# Those are done

# Next up: different graph classes: ba, random regular, geometric
# Then better visualisations



