import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from costs_and_heuristics import CostFunction, populate_threat_dict
from choices import CFn, dict_of_cost_names

plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600


def test_hesitancy_cost_fn():
    # Generate a list of costs to visualize distribution
    n_samples = 100000
    graph = nx.Graph()
    graph.add_nodes_from(range(1, n_samples + 1))
    hesitancies = [CostFunction(something) for something in dict_of_cost_names.keys()]

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

        plt.title(f'Distribution of Defense Costs for {dict_of_cost_names[hesitancy.function]} hesitancy function')
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
