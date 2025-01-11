import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from choices import dict_of_heur_names, dict_of_cost_names


def plot_helper(heuristics, cost_type, each_graph, each_param, latest_results, num_vertices, results, budget,
                violin_too=False, info=False):
    plot_start = time.time()

    # Create the plotting directory if it doesn't exist
    if not os.path.exists('plotting'):
        os.makedirs('plotting')

    colour_map = {cost_type: f'C{i}' for i, cost_type in enumerate(dict_of_heur_names)}
    # print(*colour_map, sep='\n')
    colour_map_for_sns = {dict_of_heur_names[h]: f'C{i}' for i, h in enumerate(heuristics)}
    # print(*colour_map_for_sns, sep='\n')

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
                                              label=dict_of_heur_names[guy[4]], density=True)
        handles.append(hist_patches[0])
        labels.append(dict_of_heur_names[guy[4]])

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
        mean_line = ax2.axhline(y=mean_value, color=colour, linestyle='--', linewidth=1)

    # Set title for scatter plot
    ax2.set_title('Scatter plot of results', fontsize=14)
    ax2.set_xlabel('Trial number', fontsize=12)
    ax2.set_ylabel('Number of nodes saved', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    # Distribution of costs for the cost function
    all_costs = [cost for heuristic_costs in results.values() for cost in heuristic_costs]

    # Box plot
    boxplot = sns.boxplot(y=all_costs, ax=ax3, color='lightblue')
    budget_line = ax3.axhline(y=budget, color='red', linestyle=':', linewidth=2, label='Budget')
    handles.append(budget_line)
    labels.append('Budget')

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
            heuristic = dict_of_heur_names[guy[4]]
            first_heuristic, *second_heuristic = heuristic.split('/')
            second_heuristic = second_heuristic[0] if second_heuristic else ''
            data.extend([[first_heuristic.strip(), second_heuristic.strip(), heuristic, value] for value in values])

        df = pd.DataFrame(data, columns=['First Heuristic', 'Second Heuristic', 'Full Heuristic', 'Nodes Saved'])

        # Sort the dataframe
        df = df.sort_values(['First Heuristic', 'Second Heuristic'])

        # Create the violin plot
        plt.figure(figsize=(14, 6))
        sns.violinplot(x='Full Heuristic', y='Nodes Saved', data=df,
                       hue='Full Heuristic', palette=colour_map_for_sns,
                       order=df['Full Heuristic'].unique(), legend=False)

        # Customize the plot
        plt.title(
            f'Distribution of Nodes Saved by Heuristic\n{dict_of_cost_names[cost_type]} fn. on {each_graph} graphs, {num_vertices} nodes, param {each_param}',
            fontsize=16, fontweight='bold')
        plt.xlabel('Heuristic', fontsize=14)
        plt.ylabel('Number of Nodes Saved', fontsize=14)
        plt.xticks(rotation=45, ha='right')

        # vertical lines to separate heuristic groups
        for i in [3, 6]:
            plt.axvline(x=i-0.5, color='black', linestyle=(0, (5, 10)), linewidth=1, alpha=0.5)

        # Save the violin plot to the plotting directory with a suitable filename
        violin_filename = f'plotting/violin_{each_graph.replace(" ", "_")}_{cost_type}.png'
        plt.tight_layout()
        plt.savefig(violin_filename)

        plt.close()

    return time.time() - plot_start
