import os
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt


# functions to provide graph metrics and plots

def plot_graph_and_degree_distribution(G, year=-1, filename="", title_name="", colour='mediumseagreen'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    pos = nx.spring_layout(G, seed=60, k=0.5)  # higher k, more space between nodes

    node_size = [20 * G.degree(n) for n in G.nodes()]  # scale node size by degree
    nx.draw_networkx_nodes(G, pos, node_color=colour, node_size=node_size, alpha=0.9, ax=ax1)
    nx.draw_networkx_edges(G, pos, edge_color='grey', alpha=0.7, ax=ax1)

    ax1.set_title(f'Network{f" from {year}" if year != -1 else ""}', fontsize=36, fontname='CMU Bright', weight='bold', pad=10)

    # degree rank plot
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    ax2.plot(degree_sequence, marker='o', color=colour)
    ax2.set_title('Degree Rank Plot', fontsize=36, fontname='CMU Bright', weight='bold', pad=10)
    ax2.set_xlabel('Rank', fontsize=26)
    ax2.set_ylabel('Degree', fontsize=26)
    ax2.tick_params(labelsize=20)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.suptitle(title_name, fontsize=40, fontname='CMU Bright', fontweight='bold')
    plt.subplots_adjust(top=0.85)
    plt.savefig(filename)


def main():
    master_df = pd.read_csv('network-data-in/reptilia-lizard-network-social.csv', delimiter=None, engine='python', sep=None)
    df = master_df

    G = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1])

    print(G)

    metadata = {
        'Nodes': G.number_of_nodes(),
        'Edges': G.number_of_edges(),
        'Density': nx.density(G),
        'Max degree': max(dict(G.degree()).values()),
        'Min degree': min(dict(G.degree()).values()),
        'Average degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'Assortativity': nx.degree_assortativity_coefficient(G),
        'Clustering': nx.average_clustering(G)
        # 'Diameter': nx.diameter(G)
    }

    for m in metadata:
        print(m, metadata[m])

    # example usage:

    # plot_graph_and_degree_distribution(G, filename='output/requested_plots/malawi-pilot-network.png',
    #                                    title_name='Rural Malawian Villagers Interaction Network',
    #                                    colour='mediumseagreen')


if __name__ == '__main__':
    main()

