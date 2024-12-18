import math
import time

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

plt.rc('font', family='serif')
plt.rc('text', usetex=True)


def cost_firefighter(graph, burning, protected, choose, cost_function, num_rounds, INFO=False, budget=5):
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
        costs = cost_function(graph, protected, burning, threat_dict=threat_dict)
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


def uniform_cost(graph, protected, burning, value=1, threat_dict={}):
    dict_of_costs = {}
    for vertex in graph.nodes():
        dict_of_costs[vertex] = value
    return dict_of_costs


def uniform_random_cost(graph, protected, burning, value=1, threat_dict={}):
    dict_of_costs = {}
    max_val = 5
    for vertex in graph.nodes():
        dict_of_costs[vertex] = random.randint(1, max_val)
    return dict_of_costs


# NB Designed for HIGH budgets e.g. defending one vertex should cost on avg. 25
def hesitancy_distribution_cost(graph, protected, burning, value=1, threat_dict={}, mean=29.72, std_dev=10):
    """
    Generate a defense cost for a vertex in the firefighter problem.

    :param mean: The mean cost, average vaccine uptake (default 29.72)
    :param std_dev: The standard deviation for the normal distribution (default is 10). Should be adjusted based on
      graph density s.t. average uptake is appropriate - in original paper, 75.2% uptake of vaccines so each
    :return: An integer representing the defense cost, clamped between 1 and 100.
    """
    dict_of_costs = {}

    # if std_dev == -1:
    #     z_score = stats.norm.ppf(probability)  # Find Z-score for the given probability
    #     # Calculate required standard deviation
    #     std_dev = (budget - mean) / z_score

    for vertex in graph.nodes():
        # Generate a normally distributed cost
        init_cost = random.normalvariate(mean, std_dev)

        # Clamp the cost between 1 and 100 and add to dict
        dict_of_costs[vertex] = int(np.clip(round(init_cost), 1, 100))

    return dict_of_costs


def binary_distribution_cost(graph, protected, burning, value=1, threat_dict={}, probability=0.2972):
    """
    Generate a defense cost for a vertex in the firefighter problem.

    :param mean: The mean cost, average vaccine uptake (default 29.72)
    :param std_dev: The standard deviation for the normal distribution (default is 10). Should be adjusted based on
      graph density s.t. average uptake is appropriate - in original paper, 75.2% uptake of vaccines so each
    :return: An integer representing the defense cost, clamped between 1 and 100.
    """
    dict_of_costs = {}
    for vertex in graph.nodes():
        # Generate a random number
        r = random.random()

        # costs 1 if r above mean, 0 otherwise
        dict_of_costs[vertex] = 1 if r > probability else 0

    return dict_of_costs


def test_hesitancy_cost_fn():
    # Generate a list of costs to visualize distribution
    n_samples = 100000
    costs = [hesitancy_distribution_cost(_, _, _) for _ in range(n_samples)]
    # Analyze and print results
    print(f"Mean Cost: {np.mean(costs):.2f}")
    print(f"Median Cost: {np.median(costs):.2f}")
    print(f"Min Cost: {min(costs)}")
    print(f"Max Cost: {max(costs)}")
    plt.figure(figsize=(12, 6))
    plt.hist(costs, bins=(max(costs) - min(costs)), edgecolor='black', alpha=0.7)
    plt.title('Distribution of Defense Costs')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(costs), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(costs):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# test_hesitancy_cost_fn()


def threat_cost(graph, protected, burning, value=1, threat_dict={}):
    dict_of_costs = {}
    for vertex in threat_dict:
        dict_of_costs[vertex] = threat_dict[vertex]
    return dict_of_costs


def stochastic_threat_cost(graph, protected, burning, threat_dict, value=1, spread=1):
    dict_of_costs = {}
    for vertex in threat_dict:
        dict_of_costs[vertex] = threat_dict[vertex] + random.randint(-spread, spread)
        if dict_of_costs[vertex] < 0:
            dict_of_costs[vertex] = 1
    # print(dict_of_costs)
    return dict_of_costs


def stochastic_threat_cost_low(graph, protected, burning, threat_dict, value=1):
    return stochastic_threat_cost(graph, protected, burning, threat_dict, spread=1)


def stochastic_threat_cost_mid(graph, protected, burning, threat_dict, value=1):
    return stochastic_threat_cost(graph, protected, burning, threat_dict, spread=2)


def stochastic_threat_cost_hi(graph, protected, burning, threat_dict, value=1):
    return stochastic_threat_cost(graph, protected, burning, threat_dict, spread=3)


def random_choice(graph, protected, burning, costs, threat_dict={}):
    open = graph.nodes() - protected - burning
    return random.choice(list(open))


def cheapest_choice(graph, protected, burning, costs, threat_dict={}):
    open = graph.nodes() - protected - burning
    return min(open, key=lambda x: costs[x])


def calculate_threat(graph, protected, burning, cost_function):
    dict_threat = {}
    open = graph.nodes() - protected - burning
    for vertex in open:
        current_shortest = len(open)
        dist = graph.number_of_edges()
        for fire in burning:
            if nx.has_path(graph, fire, vertex):
                dist = nx.shortest_path_length(graph, fire, vertex)
                if dist < current_shortest:
                    current_shortest = dist
        dict_threat[vertex] = dist
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
    degree_cheapest = {}
    # print ("OPEN IS "  + str(open))
    for v in costs:
        if v in open:
            degree_cheapest[v] = (graph.degree(v), costs[v])
    return min(degree_cheapest.keys(), key=lambda x: degree_cheapest[x])


# Example usage


# Now we want some basic experiments.
# start with: on graph type (start with ER), for each graph, cost function, heuristic how many are saved?


def run_experiments(size, p, num_trials, graph_type, cost_type=uniform_cost, budget=5):
    trials = {}
    # for graph in graph_types:
    dict_of_choice_funs = {"degree_cheapest": degree_cheapest, "degree_choice": degree_choice,
                           "threat_cheapest_cost": threat_cheapest,
                           "cost_only_choice": cheapest_choice}
    #  , 'stochastic_threat_cost': stochastic_threat_cost}
    # dict_of_cost_funs = {'threat_cost': threat_cost}
    print('-'*num_trials, f'100% ({num_trials} trials)')
    for i in range(num_trials):
        print("-", end="")
        if graph_type.lower() in ['erdos renyi', 'er']:
            g = nx.barabasi_albert_graph(size, p)
        elif graph_type.lower() in ['barabasi-albert', 'barabasi albert', 'ba']:
            g = nx.barabasi_albert_graph(m=p, n=size)
        elif graph_type.lower() in ['random geometric', 'rand geom', 'geometric random', 'geom rand']:
            g = nx.random_geometric_graph(radius=p, n=size)
        elif ' ' in graph_type and graph_type.split()[0] in ['random'] and graph_type.split()[1][1:] == "-regular":
            if p * size % 2 == 0:
                g = nx.random_regular_graph(d=p, n=size)  # we take p to be n in 'random n-regular'
            else:
                print(f'\ngenerating random {p}-regular with {size + 1} vertices rather than {size} \n'
                      f'  since num vertices * d must be even for random d-reg graph')
                g = nx.random_regular_graph(d=p, n=(size + 1))
        else:
            raise Exception(f"\nSorry, I didn't recognise this graph type: {graph_type}")

        # for cost_type in dict_of_cost_funs:
        for choice_type in dict_of_choice_funs:
            saved = cost_firefighter(g, {0}, set([]), dict_of_choice_funs[choice_type], cost_type,
                                     num_rounds=int(g.number_of_nodes()/2), budget=budget)
            if (graph_type, size, p, cost_type, choice_type) not in trials:
                trials[(graph_type, size, p, cost_type, choice_type)] = []
            trials[(graph_type, size, p, cost_type, choice_type)].append(saved)
    print(" 100% - complete")
    return trials


def do_the_experiments(dict_of_cost_funs, dict_of_cost_names, dict_of_choice_names, dict_of_graph_types, colour_map,
                       num_vertices=100, budget=5, num_trials=50, info=False):
    start = time.time()
    for cost_type in dict_of_cost_funs:
        each_cost_start = time.time()

        for each_type in dict_of_graph_types.keys():
            each_graph_start = time.time()
            results = {}
            for each_param in dict_of_graph_types[each_type]:
                print(f'Cost fn.: {cost_type}, graph type: {each_type}, input param.: {each_param}')
                latest_results = run_experiments(num_vertices, each_param, num_trials, each_type,
                                                 cost_type=dict_of_cost_funs[cost_type], budget=budget)
                results.update(latest_results)
                plt.clf()
                for guy in latest_results:
                    colour = colour_map.get(guy[4], 'gray')  # Default to gray if not found
                    if info:
                        print(guy, results[guy])

                    # Histogram
                    plt.subplot(1, 2, 1)  # First subplot for histogram
                    counts, bins, _ = plt.hist(results[guy], bins=30, alpha=0.4,
                                               label=dict_of_choice_names[guy[4]], density=True)
                    plt.xlim(-5, num_vertices+5)
                    plt.ylim(0, min(counts.max() * 1.2, 1))
                    plt.ylabel('Frequency')
                    plt.xlabel('Number of nodes saved')
                    plt.title('Histogram of results')
                    plt.legend()

                    # Scatter Plot
                    plt.subplot(1, 2, 2)  # Second subplot for scatter plot
                    x_values = np.arange(len(results[guy]))  # X-axis: trial number or index

                    # Scatter points for the heuristic
                    scatter = plt.scatter(x_values, results[guy], alpha=0.6,
                                          label=dict_of_choice_names[guy[4]], color=colour)

                    # Mean line for the heuristic
                    mean_value = np.mean(results[guy])
                    mean_line = plt.axhline(y=mean_value, color=colour, linestyle='--',
                                            label=f'Mean ({dict_of_choice_names[guy[4]]})')

                # Add legends to both subplots after all plotting is done
                plt.subplot(1, 2, 1)  # Go back to histogram subplot
                plt.legend(loc='upper right')

                plt.subplot(1, 2, 2)  # Go to scatter plot subplot
                plt.legend(loc='upper right')

                # Show the combined plots
                plt.title(
                    f'{dict_of_cost_names[cost_type]} fn.\non {each_type} graphs on\n{num_vertices} vertices, '
                    f'input param {each_param}')
                plt.tight_layout()
                plt.show()
                each_graph_end = time.time() - each_graph_start
                print(
                    f'end of considering {each_type} with {num_vertices} vx, param {each_param}, under {cost_type}, '
                    f'took {each_graph_end} secs')

        each_cost_end = time.time() - each_cost_start
        print(f'end of considering {cost_type}, took {each_cost_end} secs')
    print('All done!')
    print(f'TOTAL time taken: {time.time()-start} secs')

def main():
    # dict_of_cost_funs = {'stochastic_threat_cost_low': stochastic_threat_cost_low,
    #                      'stochastic_threat_cost_mid': stochastic_threat_cost_mid,
    #                      'stochastic_threat_cost_hi': stochastic_threat_cost_hi,
    #                      'uniform_random_cost': uniform_random_cost,
    #                      'uniform_cost': uniform_cost, }
    # dict_of_cost_names = {'stochastic_threat_cost_low': 'Low-stochasticity threat',
    #                       'stochastic_threat_cost_mid': 'Mid-stochasticity threat',
    #                       'stochastic_threat_cost_hi': 'High-stochasticity threat',
    #                       'uniform_random_cost': 'Uniform random',
    #                       'uniform_cost': 'Uniform Cost'}

    choices = ["degree_cheapest", "degree_choice", "threat_cheapest_cost", "cost_only_choice"]

    dict_of_choice_names = {choices[0]: 'Degree/cost', choices[1]: 'Degree',
                            choices[2]: 'Threat/cost',
                            choices[3]: 'Cost'}
    num_vertices = 100
    num_trials = 50

    # {name of graph type: [[num vertices, probability]]}
    graph_types = {"random geometric": [0.1, 0.15, 0.2],
                   "random n-regular": [3, 4, 5],
                   "Barabasi-Albert": [int(num_vertices / 16), int(num_vertices / 8)]}

    colour_map = {choices[i]: f'C{i}' for i in range(len(choices))}

    dict_of_cost_funs = {'hesitancy_cost': hesitancy_distribution_cost, 'binary_cost': binary_distribution_cost}
    dict_of_cost_names = {'hesitancy_cost': 'Hesitancy cost', 'binary_cost': 'Binary cost'}

    do_the_experiments(dict_of_cost_funs, dict_of_cost_names, dict_of_choice_names, graph_types, colour_map,
                       num_vertices=num_vertices, budget=15, num_trials=num_trials)


if __name__ == main():
    main()

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
