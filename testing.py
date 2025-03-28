import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from costs_and_heuristics import CostFunction, populate_threat_dict, CFn, Heuristic, HeuristicChoices

plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600


# Tests used during development
def test_hesitancy_cost_fn():
    # Generate a list of costs to visualize distribution
    n_samples = 100000
    graph = nx.Graph()
    graph.add_nodes_from(range(1, n_samples + 1))

    hesitancies = [CostFunction(cfn) for cfn in CFn]

    for hesitancy in hesitancies:
        costs = list(hesitancy.cost(graph, populate_threat_dict(graph, burning={1}, protected=set())).values())

        # Analyze and print results
        print(f"Mean Cost: {np.mean(costs):.2f}")
        print(f"Median Cost: {np.median(costs):.2f}")
        print(f"Min Cost: {min(costs)}")
        print(f"Max Cost: {max(costs)}")

        plt.figure(figsize=(12, 6))

        # Get unique cost values and sort them
        unique_costs = sorted(np.unique(costs))

        # Create histogram with bins centered on unique costs
        hist, bin_edges = np.histogram(costs, bins=len(unique_costs),
                                       range=(min(unique_costs) - 0.5, max(unique_costs) + 0.5))

        # Plot the histogram
        plt.bar(unique_costs, hist, width=0.8, edgecolor='black', alpha=0.7, label='Costs')

        # Set x-ticks to unique cost values
        plt.xticks(unique_costs, [f'{cost:.2f}' for cost in unique_costs])

        plt.title(f'Distribution of Defense Costs for {hesitancy.function} hesitancy function')
        plt.xlabel('Cost')
        plt.ylabel('Frequency')

        legend_elements = []
        if hesitancy.function is not CFn.HESITANCY_BINARY:
            mean_line = plt.axvline(float(np.mean(costs)), color='r', linestyle='dashed', linewidth=2,
                                    label=f'Mean: {np.mean(costs):.2f}')
            legend_elements.append(mean_line)

        if legend_elements:
            # Filter out any elements with labels starting with underscore
            filtered_elements = [el for el in legend_elements + [plt.gca().patches[0]] if
                                 not el.get_label().startswith('_')]
            plt.legend(handles=filtered_elements)
        elif hesitancy.function is CFn.HESITANCY_BINARY:
            plt.legend()

        plt.grid(True, alpha=0.3)
        plt.show()


test_hesitancy_cost_fn()


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
    print('DEGREE:', Heuristic(HeuristicChoices.DEGREE).choose(g, protecc, burn, dict_costs, threat_dict))
    print('DEGREE+THREAT:',
          Heuristic(HeuristicChoices.DEGREE, HeuristicChoices.THREAT).choose(g, protecc, burn, dict_costs, threat_dict))
    print('DEGREE+COST:',
          Heuristic(HeuristicChoices.DEGREE, HeuristicChoices.COST).choose(g, protecc, burn, dict_costs, threat_dict))
    print("-----")
    print('THREAT:', Heuristic(HeuristicChoices.THREAT).choose(g, protecc, burn, dict_costs, threat_dict))
    print('THREAT+DEGREE:',
          Heuristic(HeuristicChoices.THREAT, HeuristicChoices.DEGREE).choose(g, protecc, burn, dict_costs, threat_dict))
    print('THREAT+COST:',
          Heuristic(HeuristicChoices.THREAT, HeuristicChoices.COST).choose(g, protecc, burn, dict_costs, threat_dict))
    print("-----")
    print('COST:', Heuristic(HeuristicChoices.COST).choose(g, protecc, burn, dict_costs, threat_dict))
    print('COST+DEGREE',
          Heuristic(HeuristicChoices.COST, HeuristicChoices.DEGREE).choose(g, protecc, burn, dict_costs, threat_dict))
    print('COST+THREAT:',
          Heuristic(HeuristicChoices.COST, HeuristicChoices.THREAT).choose(g, protecc, burn, dict_costs, threat_dict))


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
        plt.hist(list(costs.values()), label=str(c).split('.')[1].replace('_', ' '), bins=max(costs.values()),
                 alpha=0.4, density=True)
        plt.legend()
        plt.show()


# test_heuristic()
# test_cost_function()
