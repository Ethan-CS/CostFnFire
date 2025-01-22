import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator

from costs_and_heuristics import HeuristicChoices


def plot_helper(heuristics, cost_type, each_graph, each_param, latest_results, num_vertices, results, budget, costs,
                degrees, violin_too=False, info=False, location=''):
    plot_start = time.time()

    # Define base colors for each main heuristic
    base_colors = sns.color_palette("hsv", len(HeuristicChoices))
    base_color_map = {str(h): base_colors[i] for i, h in enumerate(HeuristicChoices)}

    # Generate shades for each heuristic combination
    colour_map = {}
    for h in heuristics:
        main_heuristic = str(h.which_heuristic)
        base_color = to_rgba(base_color_map[main_heuristic])
        shade_factor = 0.3 + 0.7 * (hash(str(h)) % 100) / 100  # Generate a shade factor between 0.3 and 1
        colour_map[str(h)] = (
            base_color[0] * shade_factor, base_color[1] * shade_factor, base_color[2] * shade_factor, base_color[3])

    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600

    plt.clf()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 7), width_ratios=[3, 3, 1, 1])
    handles = []
    labels = []
    # recall results formatted as (graph_type, size, p, seed, outbreaks, cost_type, choice_type): num_saved_vertices
    aggregated_results = {}
    for guy, values in latest_results.items():
        heuristic_label = str(guy[-1])
        if heuristic_label not in aggregated_results:
            aggregated_results[heuristic_label] = []
        aggregated_results[heuristic_label].extend(values)

    # Precompute histogram data
    hist_data = {label: np.histogram(values, bins=30, density=True) for label, values in aggregated_results.items()}

    for heuristic_label, values in aggregated_results.items():
        colour = colour_map.get(heuristic_label, 'gray')  # default to gray if not found

        # Histogram
        counts, bins = hist_data[heuristic_label]
        ax1.hist(bins[:-1], bins, weights=counts, alpha=0.4, color=colour, label=heuristic_label)
        handles.append(ax1.patches[-1])
        labels.append(heuristic_label)

        ax1.set_ylim(0, min(max(counts) * 1.2, 1))
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_xlabel('Number of nodes saved', fontsize=12)
        ax1.set_title('Histogram', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Scatter Plot
        x_values = np.arange(len(values))
        ax2.scatter(x_values, values, alpha=0.6, color=colour)
        mean_value = np.mean(values)  # mean line for the heuristic
        ax2.axhline(y=mean_value, color=colour, linestyle='--', linewidth=1)
        ax2.set_title('Scatter plot', fontsize=14)
        ax2.set_xlabel('Trial number', fontsize=12)
        ax2.set_ylabel('Nodes saved', fontsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Box plot for costs
    sns.boxplot(y=costs, ax=ax3, color='lightblue')
    budget_line = ax3.axhline(y=budget, color='red', linestyle=':', linewidth=2, label='Budget')
    handles.append(budget_line)
    labels.append('Budget')

    ax3.set_title('Distribution of costs', fontsize=14)
    ax3.set_ylabel('Node cost', fontsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Box plot for degree distribution
    sns.boxplot(y=degrees, ax=ax4, color='lightgreen')
    ax4.set_title('Degree distribution', fontsize=14)
    ax4.set_ylabel('Degree', fontsize=12)
    ax4.tick_params(axis='both', which='major', labelsize=10)
    ax4.yaxis.set_major_locator(MaxNLocator(integer=True))

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
        elif each_graph == "tnet_malawi_pilot":
            title = f'{str(cost_type)} costs on an interaction network \n' \
                    f'from rural Malawi on {num_vertices} nodes '
        elif each_graph == "reptilia-tortoise-network-fi":
            title = f'{str(cost_type)} costs on an interaction network \n' \
                    f'of desert tortoises in Nevada in {each_param} on {num_vertices} nodes '
        else:
            title = f'{str(cost_type)} costs on {each_graph} networks \n' \
                    f'on {num_vertices} nodes with input param. {each_param}'

    plt.subplots_adjust(wspace=0.3, right=0.85, top=0.85)  # spacing between subplots

    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=10)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # save the figure to the plotting directory with a suitable filename
    filename = f'{location}/{each_graph.replace(" ", "_")}/full_{num_vertices}_{each_param}_{budget}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.savefig(filename)
    plt.close(fig)

    if violin_too:
        do_violin(cost_type, each_graph, each_param, colour_map, location, num_vertices, latest_results, budget)

    return time.time() - plot_start


def do_violin(cost_type, each_graph, each_param, colour_map, location, num_vertices, latest_results, budget):
    # Prepare data for seaborn
    data = []
    for guy, values in latest_results.items():
        data.extend([[str(guy[-1]), value] for value in values])
    df = pd.DataFrame(data, columns=['Full Heuristic', 'Nodes Saved'])
    # Sort the dataframe
    df = df.sort_values(['Full Heuristic'])

    # Create subplots for each primary heuristic
    primary_heuristics = [str(h) for h in HeuristicChoices]
    num_plots = len(primary_heuristics)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 10), sharey=True)

    for i, ph in enumerate(primary_heuristics):
        ax = axes[i] if num_plots > 1 else axes
        subset_df = df[df['Full Heuristic'].str.startswith(ph)]
        sns.violinplot(x='Full Heuristic', y='Nodes Saved', data=subset_df,
                       hue='Full Heuristic', palette=colour_map,
                       order=subset_df['Full Heuristic'], ax=ax)
        ax.set_title(f'{ph}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Heuristic', fontsize=14)
        ax.set_ylabel('Number of Nodes Saved', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    # Customize the overall plot
    if each_graph == "tnet_malawi_pilot":
        title = f'{str(cost_type)} costs on an interaction network \n' \
                f'from rural Malawi on {num_vertices} nodes '
    elif each_graph == "reptilia-tortoise-network-fi":
        title = f'{str(cost_type)} costs on an interaction network \n' \
                f'of desert tortoises in Nevada on {num_vertices} nodes '
    else:
        title = f'Distribution of Nodes Saved by Heuristic\n{str(cost_type)} fn. on {each_graph} graphs, {num_vertices} nodes, param {each_param}'

    violin_filename = f'{location}/{each_graph.replace(" ", "_")}/violin_{num_vertices}_{each_param}_{budget}.png'
    os.makedirs(os.path.dirname(violin_filename), exist_ok=True)
    fig.suptitle(title, fontsize=20, fontweight='bold')
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.savefig(violin_filename)
    plt.close()
