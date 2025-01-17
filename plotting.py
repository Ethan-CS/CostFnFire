import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_helper(heuristics, cost_type, each_graph, each_param, latest_results, num_vertices, results, budget, costs,
                violin_too=False, info=False, location=''):
    plot_start = time.time()

    colour_map = {h: f'C{i}' for i, h in enumerate(heuristics)}

    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600

    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 7), width_ratios=[3, 3, 1])
    handles = []
    labels = []
    for guy in latest_results:
        colour = colour_map.get(guy[4], 'gray')  # default to gray if not found
        if info:
            print("   ", guy, results[guy])

        # Histogram
        counts, bins, hist_patches = ax1.hist(results[guy], bins=30, alpha=0.4, color=colour,
                                              label=str(guy[4]), density=True)
        handles.append(hist_patches[0])
        labels.append(str(guy[4]))

        # ax1.set_xlim(-5, num_vertices + 5)
        ax1.set_ylim(0, min(counts.max() * 1.2, 1))
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_xlabel('Number of nodes saved', fontsize=12)
        ax1.set_title('Histogram of results', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=10)

        # Scatter Plot
        x_values = np.arange(len(results[guy]))
        scatter = ax2.scatter(x_values, results[guy], alpha=0.6, color=colour)

        # Mean line for the heuristic
        mean_value = np.mean(results[guy])
        mean_line = ax2.axhline(y=mean_value, color=colour, linestyle='--', linewidth=1)

    # Set title for scatter plot
    ax2.set_title('Scatter plot of results', fontsize=14)
    ax2.set_xlabel('Trial number', fontsize=12)
    ax2.set_ylabel('Number of nodes saved', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    # Box plot
    boxplot = sns.boxplot(y=costs, ax=ax3, color='lightblue')
    budget_line = ax3.axhline(y=budget, color='red', linestyle=':', linewidth=2, label='Budget')
    handles.append(budget_line)
    labels.append('Budget')

    ax3.set_title('Distribution of costs', fontsize=14)
    ax3.set_ylabel('Node cost', fontsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=10)

    # Set overall title based on graph type
    title = ""
    if isinstance(each_graph, str):
        if each_graph.lower() == "barabasi-albert":
            title = f'{str(cost_type)} costs on {each_graph} networks \n' \
                    f'on {num_vertices} nodes and {each_param} edges'
        elif each_graph.lower() == "random geometric":
            title = f'{str(cost_type)} costs on {each_graph} networks \n' \
                    f'on {num_vertices} nodes with radius {each_param}'
        elif each_graph.lower() == "random n-regular":
            title = f'{str(cost_type)} costs on {each_graph.replace("n-", f"{each_param}-")} networks \n' \
                    f'on {num_vertices} nodes '
        elif each_graph == "network-data-in/tnet_malawi_pilot2.csv":
            title = f'{str(cost_type)} costs on an interaction network \n' \
                    f'from rural Malawi on {num_vertices} nodes '
        elif each_graph == "network-data-in/reptilia-tortoise-network-fi.csv":
            title = f'{str(cost_type)} costs on an interaction network \n' \
                    f'of desert tortoises in Nevada in {each_param} on {num_vertices} nodes '
        else:
            title = f'{str(cost_type)} costs on {each_graph} networks \n' \
                    f'on {num_vertices} nodes with input param. {each_param}'

    plt.subplots_adjust(wspace=0.3, right=0.85, top=0.85)  # spacing between subplots
    # plt.tight_layout()

    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=10)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # save the figure to the plotting directory with a suitable filename
    if each_graph == "network-data-in/reptilia-tortoise-network-fi.csv":
        filename = f'{location}/desert-tortoise_{cost_type}_{each_param}.png'
    elif each_graph == "network-data-in/tnet_malawi_pilot2.csv":
        filename = f'{location}/malawi-village_{cost_type}.png'
    else:
        filename = f'{location}/{each_graph.replace(" ", "_")}.png'

    plt.savefig(filename)
    plt.close(fig)

    if violin_too:
        # Prepare data for seaborn
        data = []
        for guy, values in results.items():
            data.extend([[str(guy[4]), value] for value in values])

        df = pd.DataFrame(data, columns=['Full Heuristic', 'Nodes Saved'])

        # Sort the dataframe
        df = df.sort_values(['Full Heuristic'])

        # Create the violin plot
        plt.figure(figsize=(14, 6))
        sns.violinplot(x='Full Heuristic', y='Nodes Saved', data=df,
                       hue='Full Heuristic', palette={str(h): f'C{i}' for i, h in enumerate(heuristics)},
                       order=df['Full Heuristic'], legend=False)

        # Customize the plot
        if each_graph == "network-data-in/tnet_malawi_pilot2.csv":
            title = f'{str(cost_type)} costs on an interaction network \n' \
                    f'from rural Malawi on {num_vertices} nodes '
            violin_filename = f'{location}/violin_malawi_{cost_type}.png'
        elif each_graph == "network-data-in/reptilia-tortoise-network-fi.csv":
            title = f'{str(cost_type)} costs on an interaction network \n' \
                    f'of desert tortoises in Nevada on {num_vertices} nodes '
            violin_filename = f'{location}/violin_desert-tortoises_{cost_type}.png'
        else:
            title = f'Distribution of Nodes Saved by Heuristic\n{str(cost_type)} fn. on {each_graph} graphs, {num_vertices} nodes, param {each_param}'
            violin_filename = f'{location}/violin_{each_graph.replace(" ", "_")}_{cost_type}.png'
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Heuristic', fontsize=14)
        plt.ylabel('Number of Nodes Saved', fontsize=14)
        plt.xticks(rotation=45, ha='right')

        # vertical lines to separate heuristic groups
        for i in [3, 6]:
            plt.axvline(x=i - 0.5, color='black', linestyle=(0, (5, 10)), linewidth=1, alpha=0.5)

        # Save the violin plot to the plotting directory with a suitable filename

        plt.tight_layout()
        plt.savefig(violin_filename)
        plt.close()

    return time.time() - plot_start
