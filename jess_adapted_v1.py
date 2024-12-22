import os
import time

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

import junk
from choices import Heur, CFn, dict_of_cost_names
from cost_function import CostFunction
from costs_and_heuristics import Heuristic, populate_threat_dict, heuristic_names


def run_experiments(size, p, num_trials, graph_type, heuristics, cost_type, budget=5, progress=True):
    trials = {}
    # for graph in graph_types:
    choice_functions = [Heuristic(name) for name in heuristics]

    if progress:
        print(f'{junk.Colours.GREEN}-' * num_trials, f'100% ({num_trials} trials)')
    for i in range(num_trials):
        if progress:
            print("-", end="")
        g = get_graph(graph_type, p, size)

        for choice_type in choice_functions:
            saved = cost_firefighter(g, {0}, set([]), choice_type, cost_type,
                                     num_rounds=int(g.number_of_nodes() / 2), budget=budget, INFO=False)
            if (graph_type, size, p, cost_type, choice_type.which_heuristic) not in trials:
                trials[(graph_type, size, p, cost_type, choice_type.which_heuristic)] = []
            trials[(graph_type, size, p, cost_type, choice_type.which_heuristic)].append(saved)
    if progress:
        print(f" 100% - {junk.Colours.UNDERLINE}complete")
        print(f'{junk.Colours.END}')
    return trials


def get_graph(graph_type, p, size):
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
            print(f'\ngenerating random {p}-regular with {size + 1} nodes rather than {size} \n'
                  f'  since num nodes * d must be even for random d-reg graph')
            g = nx.random_regular_graph(d=p, n=(size + 1))
    else:
        raise Exception(f"\nSorry, I didn't recognise this graph type: {graph_type}")
    return g


def cost_firefighter(graph, burning, protected, heuristic_choice, cost_function_type, num_rounds=50, INFO=True,
                     budget=5):
    heuristic = Heuristic(heuristic_choice)
    if INFO:
        print("\n\n\nNEW RUN")
    total_nodes = len(graph.nodes())

    last_burning = 0
    for _ in range(num_rounds):
        open = graph.nodes() - protected - burning
        if len(open) == 0:
            break
        # choose something to protect
        cost_fn = CostFunction(cost_function_type)
        threat_dict = populate_threat_dict(graph, burning, protected)
        costs = cost_fn.cost(graph, threat_dict=threat_dict)
        # if INFO:
        print("Threat dict", threat_dict)
        print("COST FUNCTION")
        print(cost_fn, cost_function_type)
        print(costs)

        total_cost = 0
        while total_cost < budget:
            open = graph.nodes() - protected - burning
            if len(open) == 0:
                break
            node_to_protect = heuristic.choose(graph, protected, burning, costs)
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


def main():
    cost_functions = [CFn.STOCHASTIC_THREAT,
                      CFn.HESITANCY_BINARY,
                      CFn.UNIFORMLY_RANDOM]

    heuristics = [Heur.DEGREE, Heur.THREAT, Heur.COST,
              (Heur.DEGREE, Heur.THREAT),
              (Heur.DEGREE, Heur.COST),
              (Heur.THREAT, Heur.COST),
              (Heur.THREAT, Heur.DEGREE),
              (Heur.COST, Heur.DEGREE),
              (Heur.COST, Heur.THREAT)]

    num_vertices = 50
    num_trials = 1
    budget = int(num_vertices / 10)

    graph_types = {"random geometric": [0.15],
                   "random n-regular": [4],
                   "Barabasi-Albert": [int(num_vertices / 16)]}

    start = time.time()
    for cost_type in cost_functions:
        each_cost_start = time.time()

        for each_graph in graph_types.keys():
            each_graph_start = time.time()
            results = {}
            for each_param in graph_types[each_graph]:
                print(f'Cost fn.: {cost_type}, graph type: {each_graph}, input param.: {each_param}')
                latest_results = run_experiments(num_vertices, each_param, num_trials, each_graph, heuristics,
                                                 cost_type=cost_type, budget=budget)
                results.update(latest_results)
                each_graph_end = time.time() - each_graph_start
                print(
                    f'end of considering {each_graph} with {num_vertices} vx, param {each_param}, under {cost_type}, '
                    f'took {each_graph_end} secs - now plotting...'
                )
                plot_time = plot_helper(heuristics, cost_type, each_graph, each_param, latest_results, num_vertices,
                                        results, violin_too=True)
                print(f'(took {plot_time}s to plot)')
        each_cost_end = time.time() - each_cost_start
        print(f'end of considering {cost_type}, took {each_cost_end} secs')
    print('All done!')
    print(f'TOTAL time taken: {time.time() - start} secs')


def plot_helper(heuristics, cost_type, each_graph, each_param, latest_results, num_vertices, results, violin_too=False,
                info=False):
    plot_start = time.time()

    # Create the plotting directory if it doesn't exist
    if not os.path.exists('plotting'):
        os.makedirs('plotting')

    colour_map = {heuristics[i]: f'C{i}' for i in range(len(heuristics))}
    colour_map_for_sns = {heuristic_names[h]: f'C{i}' for i, h in enumerate(heuristics)}

    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600

    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), width_ratios=[3, 3, 1])  # Slightly increased width

    handles = []
    labels = []

    for guy in latest_results:
        colour = colour_map.get(guy[4], 'gray')  # Default to gray if not found
        if info:
            print("   ", guy, results[guy])

        # Histogram
        counts, bins, hist_patches = ax1.hist(results[guy], bins=30, alpha=0.4, color=colour,
                                              label=heuristic_names[guy[4]], density=True)
        handles.append(hist_patches[0])
        labels.append(heuristic_names[guy[4]])

        ax1.set_xlim(-5, num_vertices + 5)
        ax1.set_ylim(0, min(counts.max() * 1.2, 1))
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_xlabel('Number of nodes saved', fontsize=12)
        ax1.set_title('Histogram of results', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=10)

        # Scatter Plot
        x_values = np.arange(len(results[guy]))  # X-axis: trial number or index
        scatter = ax2.scatter(x_values, results[guy], alpha=0.6, color=colour)

        # Mean line for the heuristic
        mean_value = np.mean(results[guy])
        mean_line = ax2.axhline(y=mean_value, color=colour, linestyle='--')

    # Set title for scatter plot
    ax2.set_title('Scatter plot of results', fontsize=14)
    ax2.set_xlabel('Trial number', fontsize=12)
    ax2.set_ylabel('Number of nodes saved', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    # Distribution of costs for the cost function
    all_costs = [cost for heuristic_costs in results.values() for cost in heuristic_costs]

    # Box plot
    sns.boxplot(y=all_costs, ax=ax3, color='lightblue')

    ax3.set_title('Distribution of costs', fontsize=14)
    ax3.set_ylabel('Node cost', fontsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=10)

    # Set overall title based on graph type
    if each_graph.lower() == "barabasi-albert":
        title = f'{dict_of_cost_names[cost_type]} costs on {each_graph} graphs \n' \
                f'on {num_vertices} nodes and {each_param} edges'
    elif each_graph.lower() == "random geometric":
        title = f'{dict_of_cost_names[cost_type]} costs on {each_graph} graphs \n' \
                f'on {num_vertices} nodes with radius {each_param}'
    elif each_graph.lower() == "random n-regular":
        title = f'{dict_of_cost_names[cost_type]} costs on {each_graph.replace("n-", f"{each_param}-")} graphs \n' \
                f'on {num_vertices} nodes '
    else:
        title = f'{dict_of_cost_names[cost_type]} costs on {each_graph} graphs \n' \
                f'on {num_vertices} nodes with input param. {each_param}'

    fig.suptitle(title, fontsize=16, fontweight='bold')

    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, right=0.85)  # Increase spacing between subplots

    # Save the figure to the plotting directory with a suitable filename
    filename = f'plotting/{each_graph.replace(" ", "_")}_{cost_type}.png'
    plt.savefig(filename)
    plt.close(fig)

    if violin_too:
        # Prepare data for seaborn
        data = []
        for guy, values in results.items():
            heuristic = heuristic_names[guy[4]]
            data.extend([(heuristic, value) for value in values])

        df = pd.DataFrame(data, columns=['Heuristic', 'Nodes Saved'])

        # Create the violin plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='Heuristic', y='Nodes Saved', data=df,
                       hue='Heuristic', palette=colour_map_for_sns,
                       legend=False)

        # Customize the plot
        plt.title(
            f'Distribution of Nodes Saved by Heuristic\n{dict_of_cost_names[cost_type]} fn. on {each_graph} graphs, {num_vertices} nodes, param {each_param}',
            fontsize=16, fontweight='bold')
        plt.xlabel('Heuristic', fontsize=14)
        plt.ylabel('Number of Nodes Saved', fontsize=14)
        plt.xticks(rotation=45, ha='right')

        # Save the violin plot to the plotting directory with a suitable filename
        violin_filename = f'plotting/violin_{each_graph.replace(" ", "_")}_{cost_type}.png'
        plt.tight_layout()
        plt.savefig(violin_filename)

        # Close the figure to free up memory
        plt.close()

    return time.time() - plot_start


if __name__ == main():
    main()

# add a periodic cost function?
# a heuristic that is in-proportion stochastically to cost?
# OR even better a cost that is stochastically related to threat
# Those are done

# Better visualisations
