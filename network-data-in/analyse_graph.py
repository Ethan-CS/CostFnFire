import os
import pandas as pd
import networkx as nx


def read_csv_files(directory=os.path.dirname(os.path.abspath(__file__))):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    return csv_files


def create_graph_from_csv(file_path):
    df = pd.read_csv(file_path, delimiter=None, engine='python', sep=None)
    if df.shape[1] < 2:
        raise ValueError(f"CSV file {file_path} does not have enough columns to create a graph.")
    G = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1])
    return G


def get_graph_metadata(G):
    metadata = {
        'Nodes': G.number_of_nodes(),
        'Edges': G.number_of_edges(),
        'Density': nx.density(G),
        'Max degree': max(dict(G.degree()).values()),
        'Min degree': min(dict(G.degree()).values()),
        'Average degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'Assortativity': nx.degree_assortativity_coefficient(G)
    }
    return metadata


def main():
    for csv_file in read_csv_files():
        G = create_graph_from_csv(csv_file)
        metadata = get_graph_metadata(G)

        print(f'File: {csv_file}')
        for key, value in metadata.items():
            print(f'{key}: {value}')
        print()


if __name__ == '__main__':
    main()
