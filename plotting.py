import ast
import datetime
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba, to_hex
from matplotlib.ticker import MaxNLocator

import graph_analysis
from costs_and_heuristics import HeuristicChoices, Heuristic

plt.rcParams['font.family'] = 'CMU Bright'


# plotting functions - mostly helpers for formatting

def plot_helper(heuristics, cost_type, each_graph, each_param, latest_results, num_vertices, budget, costs,
                degrees, row_1=True, row_2=True, location=''):
    plot_start = time.time()
    base_colours, colour_map = get_colours(heuristics, extra=2)
    primary_heuristics = []
    for guy in latest_results.keys():
        if isinstance(guy[-1], Heuristic):
            primary_heuristics.append(str(guy[-1].which_heuristic))
        elif isinstance(guy[-1], str) and '/' in guy[-1]:
            primary_heuristics.append(guy[-1].split('/')[0])
        else:
            primary_heuristics.append(str(guy[-1]))

    primary_heuristics = sorted(set(primary_heuristics))

    boxplot_colours = base_colours[len(primary_heuristics):]

    plt.rcParams['figure.dpi'] = 1200
    plt.rcParams['savefig.dpi'] = 1200

    handles, labels = [], []

    aggregated_results = {}
    for guy in latest_results.keys():
        values = latest_results[guy]
        aggregated_results.setdefault(str(guy[-1]), []).extend(values)
    hist_data = {label: np.histogram(np.array(values), bins=30, density=True) for label, values in
                 aggregated_results.items()}

    if row_1:
        plt.clf()
        fig, axes = plt.subplots(1, 4, figsize=(18, 6.5))
        ax1, ax2, ax3, ax4 = axes

        for heuristic_label, values in aggregated_results.items():
            colour = colour_map.get(heuristic_label, 'gray')
            counts, bins = hist_data[heuristic_label]
            ax1.hist(bins[:-1], bins, weights=counts, alpha=0.4, color=colour, label=heuristic_label)
            handles.append(ax1.patches[-1])
            labels.append(heuristic_label)
            ax1.set_ylim(0, min(max(counts) * 1.2, 1))
            ax1.set_ylabel('Density', fontsize=12)
            ax1.set_xlabel('Number of nodes saved', fontsize=12)
            ax1.set_title('Histogram', fontsize=16, fontweight='bold')
            ax1.tick_params(axis='both', which='major', labelsize=10)
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

            x_values = np.arange(len(values))
            ax2.scatter(x_values, values, alpha=0.6, color=colour)
            mean_value = np.mean(values)
            ax2.axhline(y=mean_value, color=colour, linestyle='--', linewidth=1)
            ax2.set_title('Scatter plot', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Trial number', fontsize=12)
            ax2.set_ylabel('Nodes saved', fontsize=12)
            ax2.tick_params(axis='both', which='major', labelsize=10)
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

        sns.boxplot(y=costs, ax=ax3, color=boxplot_colours[0], width=0.4)
        budget_line = ax3.axhline(y=budget, color='red', linestyle=':', linewidth=2, label='Budget')
        handles.append(budget_line)
        labels.append('Budget')
        ax3.set_title('Distribution of costs', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Node cost', fontsize=12)
        ax3.tick_params(axis='both', which='major', labelsize=10)
        ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

        sns.boxplot(y=[degree for sublist in degrees for degree in sublist], ax=ax4, color=boxplot_colours[1],
                    width=0.4)
        ax4.set_title('Degree distribution', fontsize=16, fontweight='bold')
        ax4.set_ylabel('Degree', fontsize=12)
        ax4.tick_params(axis='both', which='major', labelsize=10)
        ax4.yaxis.set_major_locator(MaxNLocator(integer=True))

        fig.suptitle(get_title(cost_type, each_graph, each_param, num_vertices), fontsize=18, fontweight='bold', y=0.95)
        plt.subplots_adjust(wspace=0.3, hspace=0.3, right=0.85, top=0.8, bottom=0.2)
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=10)

        if each_param > -1:
            filename = f'{location}/{each_graph.replace(" ", "_")}/{location.split("/")[-1]}_hist_{num_vertices}_{each_param}_{budget}.png'
        else:
            filename = f'{location}/{each_graph.replace(" ", "_")}/{location.split("/")[-1]}_hist_{num_vertices}_{budget}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close(fig)

    if row_2:
        plt.clf()
        fig, axes = plt.subplots(1, 4, figsize=(18, 6.5))
        data = []
        for guy, values in latest_results.items():
            data.extend([[str(guy[-1]), value] for value in values])
        df = pd.DataFrame(data, columns=['Full Heuristic', 'Nodes Saved'])
        df = df.sort_values(['Full Heuristic'])

        y_min, y_max = float('inf'), float('-inf')
        for i, ph in enumerate(primary_heuristics):
            which_axis = axes[i]
            subset_df = df[df['Full Heuristic'].str.startswith(ph)]
            sns.violinplot(x='Full Heuristic', y='Nodes Saved', data=subset_df,
                           hue='Full Heuristic', palette=colour_map,
                           order=subset_df['Full Heuristic'], ax=which_axis)
            which_axis.set_title(f'{ph}', fontsize=20, fontweight='bold')
            if i == 0:
                which_axis.set_ylabel('Number of Nodes Saved', fontsize=20)
            else:
                which_axis.set_ylabel('')
            which_axis.set_xlabel('')
            which_axis.tick_params(axis='x', rotation=45)
            plt.setp(which_axis.get_xticklabels(), rotation=45, horizontalalignment='right')

            current_y_min, current_y_max = which_axis.get_ylim()
            y_min = min(y_min, current_y_min)
            y_max = max(y_max, current_y_max)

        for i, ph in enumerate(primary_heuristics):
            which_axis = axes[i]
            which_axis.set_ylim(y_min, y_max)

        fig.text(0.5, 0.02, 'Heuristic', ha='center', fontsize=16)
        fig.suptitle(get_title(cost_type, each_graph, each_param, num_vertices), fontsize=18, fontweight='bold', y=0.95)
        plt.subplots_adjust(wspace=0.3, hspace=0.3, right=0.85, top=0.8, bottom=0.2)
        if each_param > -1:
            filename = f'{location}/{each_graph.replace(" ", "_")}/{location.split("/")[-1]}_violin_{num_vertices}_{each_param}_{budget}.png'
        else:
            filename = f'{location}/{each_graph.replace(" ", "_")}/{location.split("/")[-1]}_violin_{num_vertices}_{budget}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close(fig)

    return time.time() - plot_start


def get_title(cost_type, each_graph, each_param, num_vertices):
    title = "Number of vertices saved with "
    if isinstance(each_graph, str):
        # Number of vertices saved under a threat-based cost function with added stochasticity
        # on a {graph name} on {num_vertices} {nodes or e.g. name of animal} {ehere e.g. South Australia}
        if each_graph.lower() == "barabasi-albert":
            title += f'{str(cost_type)}-based costs on {each_graph} networks \n' \
                     f'on {num_vertices} nodes and {each_param} edges'
        elif each_graph.lower() == "random geometric":
            title += f'{str(cost_type)}-based costs on {each_graph} networks \n' \
                     f'on {num_vertices} nodes with radius {each_param}'
        elif each_graph.lower() == "random n-regular":
            title += f'{str(cost_type)}-based costs on {each_graph.replace("n-", f"{each_param}-")} networks \n' \
                     f'on {num_vertices} nodes '
        elif each_graph == "tnet_malawi_pilot":
            title += f'{str(cost_type)}-based costs on an interaction network \n' \
                     f'of {num_vertices} villagers in rural Malawi'
        elif each_graph == "reptilia-tortoise-network-fi":
            title += f'{str(cost_type)}-based costs on an interaction network \n' \
                     f'of {num_vertices} desert tortoises in Nevada in {each_param}'
        elif each_graph == "mammalia-raccoon-proximity":
            title += f'{str(cost_type)}-based costs on a temporal interaction network \n' \
                     f'of {num_vertices} raccoons in suburban Illinois'
        elif each_graph == "reptilia-lizard-network-social":
            title += f'{str(cost_type)}-based costs on a social interaction network \n' \
                     f'of {num_vertices} sleepy lizards in South Australia'
        else:
            title += f'{str(cost_type)}-based costs on {each_graph} networks \n' \
                     f'on {num_vertices} nodes with input param. {each_param}'
    return title


def get_colours(heuristics, colour_palette='bright', blend_factor=0.65, adjust=0.5, extra=0, blend=False):
    # Define a color palette with more colors than the number of heuristics
    base_colors = sns.color_palette(palette=colour_palette, n_colors=len(HeuristicChoices) - 1 + extra)
    base_colors.insert(HeuristicChoices.index_of(HeuristicChoices.from_string('Random')), 'black')

    base_color_map = {str(h): base_colors[i] for i, h in enumerate(HeuristicChoices)}

    def blend_colors(color1, color2, blend_by=0.5):
        return tuple(blend_by * np.array(color1) + (1 - blend_by) * np.array(color2))

    def adjust_color(color, factor):
        """Adjust the color by a given factor."""
        color = np.array(to_rgba(color))
        color[:3] = color[:3] * factor
        color = np.clip(color, 0, 1)
        return to_hex(color)

    colour_map = {}
    for h in heuristics:
        if isinstance(h, Heuristic):
            main_heuristic = str(h.which_heuristic)
            secondary_heuristic = str(h.tie_break)
        elif isinstance(h, str) and '/' in h:
            main_heuristic, secondary_heuristic = h.split('/')
        else:
            main_heuristic, secondary_heuristic = str(h), None
        base_color = to_rgba(base_color_map[main_heuristic])

        if blend:
            if secondary_heuristic in base_color_map:
                secondary_color = to_rgba(base_color_map[secondary_heuristic])
                blended_color = blend_colors(base_color, secondary_color, blend_by=blend_factor)
            else:
                blended_color = base_color
            colour_map[str(h)] = blended_color
        else:
            if secondary_heuristic in base_color_map:
                secondary_color = HeuristicChoices.index_of(HeuristicChoices.from_string(secondary_heuristic))
                adjusted_color = adjust_color(base_color, 1 + secondary_color / (len(HeuristicChoices) + adjust))
            else:
                adjusted_color = base_color

            colour_map[str(h)] = adjusted_color

    return base_colors, colour_map


def main():
    data_where = "data/simulation_results"
    # "added_to_paper_06-02-25/20250129114224-lizards"
    directory = f"output"

    graph_names = {
        # "reptilia-lizard-network-social": {"name": "Lizard social network", "param_name": "none"},
        # "random_geometric": {"name": "Random Geometric", "param_name": "radius"},
        # "random_n-regular": {"name": "Random n-Regular", "param_name": "degree"},
        # 'mammalia-raccoon-proximity': {"name": 'Raccoon interaction network', "param_name": "none"},
        "erdos-renyi": {"name": "Erdős–Rényi", "param_name": "probability"},
        "barabasi-albert": {"name": "Barabási–Albert", "param_name": "input edges"}
    }

    # all_detailed_plots(directory, graph_names, zettel_id)

    cost_functions = [
        ('Uniformly Random', 'uniformly random'),
        ('Uniform', 'uniform'),
        ('Hesitancy-Binary', 'binary hesitation-based'),
        ('Stochastic-Threat (high)', 'threat-based with\nhigh stochasticity'),
        ('Stochastic-Threat (low)', 'threat-based with\nlow stochasticity')
    ]

    produce_summary_plots(cost_functions, f'{directory}/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}', graph_names, data_where)


def produce_summary_plots(cost_functions, directory, graph_names, data_where, same_scale=True):
    for graph_name in graph_names.keys():
        filepath = f'output/requested_plots/{directory}'

        title_name, param_name = graph_names[graph_name]["name"], graph_names[graph_name]["param_name"]
        # graph_df = pd.read_csv(f'network-data-in/{graph_name}.csv', delimiter=None, engine='python', sep=None)
        # G = nx.from_pandas_edgelist(graph_df)

        # graph_analysis.plot_graph_and_degree_distribution(G, -1, f'{filepath}/{graph_name}.png', title_name)

        # params = graph_df[param_name].unique() if param_name != "none" else [None]
        params = pd.read_csv(f'output/{data_where}/Hesitancy-Binary/{graph_name}/results.csv'.lower())['parameter'].unique()
        for param in params:
            already_set_y_label = False
            count_for_x_label = 0
            fig, axes = plt.subplots(1, len(cost_functions), figsize=(20, 6.5))
            handles, labels = [], []
            max_y = float('-inf')
            for i, (cost_dir_name, cost_name) in enumerate(cost_functions):
                cost_dir = cost_dir_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
                ax = axes[i]
                file = f'output/{data_where}/{cost_dir}/{graph_name}/results.csv'
                df = pd.read_csv(file)
                df.columns = df.columns.str.strip()
                df = df[df['parameter'] == param]

                df['num_vertices_saved'] = df['num_vertices_saved'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                df = df.explode('num_vertices_saved')
                df['num_vertices_saved'] = pd.to_numeric(df['num_vertices_saved'])

                df = df[~df['heuristic'].str.startswith('Random/') & ~df['heuristic'].str.endswith('/Random')]

                base_colors, colour_map = get_colours(df['heuristic'].unique(), colour_palette='deep',
                                                      blend_factor=0.6)

                for heuristic in df['heuristic'].unique():
                    subset = df[df['heuristic'] == heuristic]
                    line = sns.lineplot(data=subset, x='budget', y='num_vertices_saved', label=heuristic, marker='o',
                                        color=colour_map[heuristic], ax=ax, estimator=np.median)

                    handle, label = line.get_legend_handles_labels()
                    handles.extend(h for h in handle if h not in handles)
                    labels.extend(l for l in label if l not in labels)

                ax.set_title(f'{cost_name}'.capitalize(), fontsize=22, fontweight='bold')
                if count_for_x_label == 2:
                    ax.xaxis.set_label_coords(0, -0.15)
                    ax.set_xlabel('Budget', fontsize=20)
                else:
                    ax.set_xlabel('', fontsize=1)
                count_for_x_label += 1

                if not already_set_y_label:
                    ax.yaxis.set_label_coords(-0.2, 0)
                    ax.set_ylabel('Number of Vertices Saved', fontsize=20)
                    already_set_y_label = True
                else:
                    ax.set_ylabel('', fontsize=1)
                ax.grid(True)

                ax.legend().remove()
                max_y = max(max_y, ax.get_ylim()[1])

            if same_scale:
                for ax in axes:
                    ax.set_ylim(0, max_y)

            fig.suptitle(f'Heuristic performance comparison for {title_name}{f" graphs with {param_name} {param}" if param_name.lower() != "none" else ""}\n ',
                         fontsize=28, fontweight='bold', y=0.98)
            plt.subplots_adjust(wspace=0.3, hspace=0.3, right=0.85, top=0.8, bottom=0.2)

            fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=16)
            filename = f'{filepath}/{graph_name}_cost_comparison{f"_{param}" if param_name.lower() != "none" else ""}.pdf'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=1200)
            print(f'created a fig at {filename}')
            plt.close(fig)


def all_detailed_plots(directory, graph_names, zettel_id):
    for graph_name in graph_names.keys():
        title_name, param_name = graph_names[graph_name]["name"], graph_names[graph_name]["param_name"]
        graph_df = pd.read_csv(f'network-data-in/{graph_name}.csv', delimiter=None, engine='python', sep=None)
        G = nx.from_pandas_edgelist(graph_df)
        graph_analysis.plot_graph_and_degree_distribution(G, -1,
                                                          f'output/requested_plots/{graph_name.split("-")[1]}_graph.png',
                                                          title_name)
        results_file_name = f"{graph_name}/results.csv"

        # params_full = {"barabasi-albert": [5, 10, 20, 30, 40], "erdos-renyi": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]}
        params = graph_df[param_name].unique() if param_name != "none" else [None]
        # for param in params_full[graph_name]:
        for param in params:
            # output_filename = f'output/requested_plots/{graph_name.split("-")[1]}_{f"{param}_" if param is not None else ''}graph.png'
            # G = nx.from_pandas_edgelist(graph_df[graph_df[param] == param] if param is not None else graph_df)
            # graph_analysis.plot_graph_and_degree_distribution(G, -1, output_filename,
            #                                                   # f'{directory}/plots/{graph_name.split("-")[1]}_graph.png',
            #                                                   title_name)
            combo_plot = []
            for file, name in [(f'{directory}/Uniformly Random/{results_file_name}', 'uniformly random'),
                               (f'{directory}/Uniform/{results_file_name}', 'uniform'),
                               (f'{directory}/Hesitancy-Binary/{results_file_name}', 'binary hesitation-based'),
                               (f'{directory}/Stochastic-Threat (high)/{results_file_name}',
                                'threat-based and stochastic'),
                               (f'{directory}/Stochastic-Threat (low)/{results_file_name}',
                                'threat-based, slightly stochastic')]:
                df = pd.read_csv(file)
                df.columns = df.columns.str.strip()

                df['num_vertices_saved'] = df['num_vertices_saved'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                df = df.explode('num_vertices_saved')
                df['num_vertices_saved'] = pd.to_numeric(df['num_vertices_saved'])

                df = df[~df['heuristic'].str.startswith('Random/') & ~df['heuristic'].str.endswith('/Random')]

                plt.rcParams['figure.dpi'] = 1200
                plt.rcParams['savefig.dpi'] = 1200
                plt.rcParams['font.family'] = 'CMU Bright'

                fig, axes = plt.subplots(2, 2, figsize=(20, 16))
                primary_heuristics = [h for h in df['heuristic'].unique() if '/' not in h]
                interesting_heuristics = [h for h in primary_heuristics if h != 'Random']

                base_colors, colour_map = get_colours(df['heuristic'].unique(), colour_palette='deep', blend_factor=0.6)

                handles, labels = [], []

                # Plot 1
                ax = axes[0, 0]
                for heuristic in df['heuristic'].unique():
                    subset = df[df['heuristic'] == heuristic]
                    handles, labels = do_line_plot(ax, colour_map, heuristic, subset, first=True)
                ax.set_title(f'All heuristics', fontsize=28, fontweight='bold')
                ax.set_xlabel('Budget', fontsize=26, fontstyle='italic')
                ax.set_ylabel('Number of Vertices Saved', fontsize=26, fontstyle='italic')
                ax.grid(True)
                ax.legend().remove()

                # Plot 2
                ax = axes[0, 1]
                for heuristic in df['heuristic'].unique():
                    if interesting_heuristics[0] == heuristic.split('/')[0] or heuristic == 'Random':
                        subset = df[df['heuristic'] == heuristic]
                        do_line_plot(ax, colour_map, heuristic, subset)
                ax.set_title(f'{interesting_heuristics[0]} heuristic results', fontsize=30, fontweight='bold')
                ax.set_xlabel('Budget', fontsize=26, fontstyle='italic')
                ax.set_ylabel('Vertices Saved', fontsize=26, fontstyle='italic')
                ax.grid(True)
                ax.legend().remove()

                # Plot 3
                ax = axes[1, 0]
                for heuristic in df['heuristic'].unique():
                    if interesting_heuristics[1] == heuristic.split('/')[0] or heuristic == 'Random':
                        subset = df[df['heuristic'] == heuristic]
                        do_line_plot(ax, colour_map, heuristic, subset)
                ax.set_title(f'{interesting_heuristics[1]} heuristics', fontsize=30, fontweight='bold')
                ax.set_xlabel('Budget', fontsize=26, fontstyle='italic')
                ax.set_ylabel('Vertices Saved', fontsize=26, fontstyle='italic')
                ax.grid(True)
                ax.legend().remove()

                # Plot 4
                ax = axes[1, 1]
                for heuristic in df['heuristic'].unique():
                    if interesting_heuristics[2] == heuristic.split('/')[0] or heuristic == 'Random':
                        subset = df[df['heuristic'] == heuristic]
                        do_line_plot(ax, colour_map, heuristic, subset)
                ax.set_title(f'{interesting_heuristics[2]} heuristics', fontsize=30, fontweight='bold')
                ax.set_xlabel('Budget', fontsize=26, fontstyle='italic')
                ax.set_ylabel('Vertices saved', fontsize=26, fontstyle='italic')
                ax.grid(True)
                ax.legend().remove()

                for ax in axes.flat:
                    ax.set_ylim(axes[0, 0].get_ylim())

                fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=22)

                plt.tight_layout()

                # filename = f'{directory}/plots/{graph_name.split("-")[1]}_{name.replace(" ", "_")}.png'
                # filename = f'output/requested_plots/{zettel_id}/{graph_name.split("-")[1]}_{name.replace(" ", "_")}{f"_{param}" if param is not None else ""}.png'
                filename = f'output/requested_plots/{zettel_id}-2/{graph_name}_{name.replace(" ", "_")}{f"_{param}" if param is not None else ""}.png'

                os.makedirs(os.path.dirname(filename), exist_ok=True)

                plt.suptitle(
                    f'Heuristic performance comparison for {name} costs\non a {title_name.lower()} with {param_name} {param}',
                    fontsize=36,
                    fontweight='bold')
                plt.subplots_adjust(top=0.87, right=0.82, wspace=0.3, hspace=0.3)
                plt.savefig(filename)
                plt.close(fig)


def do_line_plot(ax, colour_map, heuristic, subset, first=False):
    sns.lineplot(data=subset, x='budget', y='num_vertices_saved', label=heuristic, marker='o',
                 color=colour_map[heuristic], ax=ax, estimator=np.median)
    handles, labels = None, None
    if first:
        handles, labels = ax.get_legend_handles_labels()

    return handles, labels


if __name__ == '__main__':
    main()
