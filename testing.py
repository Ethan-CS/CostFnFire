import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

from choices import CFn
from cost_function import CostFunction


def test_hesitancy_cost_fn():
    # Generate a list of costs to visualize distribution
    n_samples = 100000
    graph = nx.Graph()
    graph.add_nodes_from(range(1, n_samples + 1))
    hesitancy = CostFunction(CFn.HESITANCY_BINARY)
    costs = list(hesitancy.cost(graph).values())
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
    plt.axvline(float(np.mean(costs)), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(costs):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

test_hesitancy_cost_fn()

