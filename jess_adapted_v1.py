import datetime
import os
import random
import time

import networkx as nx
import pandas as pd

import junk
from costs_and_heuristics import CostFunction, Heuristic, populate_threat_dict, CFn, HeuristicChoices
from plotting import plot_helper


def run_experiments(size, p, num_trials, graph_type, heuristics, cost_type, budget, outbreak, progress=True):
    trials = {}
    all_costs = []
    if progress:
        print(f'{junk.Colours.GREEN}-' * num_trials, f'100% ({num_trials} trials)')

    for i in range(num_trials):
        if progress:
            print("-", end="")
        g = get_graph(graph_type, p, size)
        if g.order() != size:  # e.g. read-in from csv, return actual size for plotting etc.
            size = g.order()
        for choice_type in heuristics:
            if outbreak is None:
                root = list(g.nodes)[0]
            elif outbreak in g.nodes:
                root = outbreak
            else:
                root = random.choice(list(g.nodes))
            saved = cost_firefighter(g, {root}, set([]), choice_type, cost_type,
                                     num_rounds=int(g.number_of_nodes() / 2), budget=budget, INFO=False)
            if (graph_type, size, p, cost_type, choice_type) not in trials:
                trials[(graph_type, size, p, cost_type, choice_type)] = []

            trials[(graph_type, size, p, cost_type, choice_type)].append(saved[0])
            all_costs.extend(saved[1])
    if progress:
        print(f" 100% - {junk.Colours.UNDERLINE}complete")
        print(f'{junk.Colours.END}')
    return trials, all_costs, size


def get_graph(graph_type, p, size):
    if isinstance(graph_type, nx.Graph):
        return graph_type
    elif isinstance(graph_type, str):
        if os.path.isfile(graph_type) and graph_type.lower().endswith('.csv'):
            try:
                df = pd.read_csv(graph_type, sep=None, engine='python')
                if len(df.columns) < 2:
                    raise ValueError("CSV file must have at least two columns for source and target nodes")
                df = df.rename(columns={df.columns[0]: 'source', df.columns[1]: 'target'})
                year_col = next((col for col in df.columns if col.lower() == 'year'), None)

                if year_col and p is not None and p != 0:
                    available_years = df[year_col].unique()
                    if p in available_years:
                        df = df[df[year_col] == p]
                    else:
                        raise ValueError(
                            f"The year {p} is not present in the data. Available years are: {sorted(available_years)}")
                if len(df.columns) > 2:
                    # Create graph with edge attributes if there are more columns
                    return nx.from_pandas_edgelist(df, 'source', 'target', edge_attr=True)
                else:
                    return nx.from_pandas_edgelist(df)
            except Exception as e:
                raise Exception(f"Error reading CSV file: {e}")

        if graph_type.lower() in ['erdos renyi', 'er']:
            return nx.erdos_renyi_graph(size, p)
        elif graph_type.lower() in ['barabasi-albert', 'barabasi albert', 'ba']:
            return nx.barabasi_albert_graph(n=size, m=p)
        elif graph_type.lower() in ['random geometric', 'rand geom', 'geometric random', 'geom rand']:
            return nx.random_geometric_graph(n=size, radius=p)
        elif ' ' in graph_type and graph_type.split()[0] in ['random'] and graph_type.split()[1][1:] == "-regular":
            if p * size % 2 == 0:
                return nx.random_regular_graph(d=p, n=size)
            else:
                print(f'\ngenerating random {p}-regular with {size + 1} nodes rather than {size} \n'
                      f'  since num nodes * d must be even for random d-reg graph')
                return nx.random_regular_graph(d=p, n=(size + 1))
    else:
        raise Exception(f"\nSorry, I didn't recognise this graph type: {graph_type}, type: {type(graph_type)}")


def cost_firefighter(graph, burning, protected, heuristic, cost_function_type, budget, num_rounds=50, INFO=True):
    if INFO:
        print("\n\n\nNEW RUN")
    total_nodes = len(graph.nodes())
    all_costs = []
    for i in range(num_rounds):
        open = graph.nodes() - protected - burning
        if len(open) == 0:
            break
        # choose something to protect
        cost_fn = CostFunction(cost_function_type)
        threat_dict = populate_threat_dict(graph, burning, protected)
        costs = cost_fn.cost(graph, threat_dict=threat_dict)
        all_costs.extend(list(costs.values()))
        if INFO:
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
    return len(graph.nodes()) - len(burning), all_costs


def main():
    cost_functions = [CFn.STOCHASTIC_THREAT,
                      CFn.HESITANCY_BINARY,
                      CFn.UNIFORMLY_RANDOM]

    heuristics = []
    for h1 in HeuristicChoices:
        heuristics.append(Heuristic(h1, None))
        for h2 in HeuristicChoices:
            if h1 != h2:
                could_add = Heuristic(h1, h2)
                if could_add not in heuristics:
                    heuristics.append(could_add)

    num_vertices = 50
    num_trials = 10
    budget = int(num_vertices / 10)  # TODO vary this sensibly
    outbreak = 'rand'  # can be an int ot 'rand' for random outbreak each time
    graph_types = {"network-data-in/tnet_malawi_pilot2.csv": [0],
                   # "random geometric": [0.15],
                   # "random n-regular": [4],
                   "Barabasi-Albert": [int(num_vertices / 16)],
                   "network-data-in/reptilia-tortoise-network-fi.csv": [2005, 2009, 2013]}

    # malawi data: http://www.sociopatterns.org/datasets/contact-patterns-in-a-village-in-rural-malawi/
    # info: "observational contact data collected for 86 individuals living in a village in rural Malawi. These data
    # were analyzed and published in the paper L. Ozella et al., “Using wearable proximity sensors to characterize
    # social contact patterns in a village of rural Malawi”, EPJ Data Science 10, 46 (2021)."

    # reptilia-tortoise-network-fi.csv: https://networkrepository.com/reptilia-tortoise-network-fi.php
    # data points in each year from 2005 to 2013 (inclusive)
    # warning! very big! give a year in params

    start = time.time()
    exp_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    for cost_type in cost_functions:
        path = f'output/{exp_id}/{cost_type}'
        os.makedirs(path, exist_ok=True)

        each_cost_start = time.time()

        for each_graph in graph_types.keys():
            print(f' ***** {each_graph} *****')
            each_graph_start = time.time()
            results = {}
            actual_num_vertices = num_vertices
            for each_param in graph_types[each_graph]:
                print(f'  cost fn.: {cost_type}, graph type: {each_graph}, input param.: {each_param}')
                latest_results, costs, actual_num_vertices = run_experiments(num_vertices, each_param, num_trials,
                                                                             each_graph,
                                                                             heuristics, cost_type=cost_type,
                                                                             budget=budget, outbreak=outbreak,)
                # latest results format:
                # (graph_type, size, p, cost_type, choice_type): num_saved_vertices
                results.update(latest_results)

                each_graph_end = time.time() - each_graph_start

                print(
                    f'end of considering {each_graph} under {cost_type}, took {each_graph_end} secs - now plotting...')
                plot_time = plot_helper(heuristics, cost_type, each_graph, each_param, latest_results,
                                        actual_num_vertices, results, budget, costs, violin_too=True, location=path)
                print(f'(took {plot_time}s to plot)')

            print('Writing to file...')
            graph_name = str(each_graph).split('/')[-1] if '/' in str(each_graph) else str(each_graph)
            filename = f'{path}/{graph_name}.txt'

            # Check if the file exists and is empty
            file_exists = os.path.exists(filename)
            file_empty = file_exists and os.path.getsize(filename) == 0

            with open(filename, "a") as f:
                if not file_exists or file_empty:
                    f.write("graph_type num_vertices parameter cost_function heuristic num_vertices_saved\n")

                for r in results:
                    f.write(' '.join(map(lambda x: x.strip() if isinstance(x, str) else str(x).strip(), r)) + ' ' + str(
                        results[r]).strip() + '\n')

            print(f'Finished writing to {filename}')

        each_cost_end = time.time() - each_cost_start
        print(f'end of considering {cost_type}, took {each_cost_end} secs')
    print('All done!')
    time_taken = time.time() - start
    print(f'TOTAL time taken: {datetime.timedelta(seconds=time_taken)} ({time_taken} secs)')


if __name__ == main():
    main()
