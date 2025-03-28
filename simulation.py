import csv
import datetime
import os
import random
import time

import networkx as nx
import pandas as pd

from costs_and_heuristics import CostFunction, Heuristic, populate_threat_dict, CFn, HeuristicChoices


# main simulation code used to run Cost function firefighter simulations on given input graph(s)

def run_experiments(size, p, num_trials, graph_type, heuristics, cost_type, budget, outbreak, progress=True):
    """
    Simulation code to run the firefighter problem on a given graph with a given heuristic and cost function.
    :param size: number of graph vertices
    :param p: generation parameter for graph generation, or filtering parameter for read-in graphs, if appropriate.
    :param num_trials: number of trials for each experiment (cost function-heuristic pair)
    :param graph_type: which graph to use for simulations
    :param heuristics: list of heuristics to use.
    :param cost_type: which cost type to use.
    :param budget: budget for this set of trials.
    :param outbreak: the vertex on which the fire starts.
    :param progress: whether to print a progress bar to console or not.
    :return: simulation results in a dictionary, with format:
    `(graph_type, size, p, seed, outbreaks, budget, cost_type, choice_type): num_saved_vertices`
    """
    trials, strategies, cost_mappings = {}, {}, {}
    all_degrees = []

    if progress:
        print(f'{Colours.GREEN}-' * num_trials, f'100% ({num_trials} trials)')

    for i in range(num_trials):
        if progress:
            print("-", end="")
        g, seed, degrees = get_graph(graph_type, p, size)
        if len(all_degrees) == 0 or graph_type not in [f for f in os.listdir('network-data-in') if os.path.isfile(os.path.join('network-data-in', f))]:
            all_degrees.append(degrees)
        if g.order() != size:  # e.g. read-in from csv, return actual size for plotting etc.
            size = g.order()
        for choice_type in heuristics:
            if outbreak is None:
                root = list(g.nodes)[0]
            elif outbreak in g.nodes:
                root = outbreak
            else:  # e.g. root is 'rand,' indicating choose a random vertex each time
                root = random.choice(list(g.nodes))

            input_params = (graph_type, size, p, seed, root, budget, cost_type, choice_type)
            trials.setdefault(input_params, [])
            strategies.setdefault(input_params, [])
            cost_mappings.setdefault(input_params, [])

            saved, strategy, cost_mapping = cost_firefighter(g, {root}, set([]), choice_type, cost_type,
                                                             num_rounds=int(g.number_of_nodes() / 2), budget=budget,
                                                             info=False)

            trials[input_params].append(saved)
            strategies[input_params].append(strategy)
            cost_mappings[input_params].append(cost_mapping)
    if progress:
        print(f" 100% - {Colours.UNDERLINE}complete")
        print(f'{Colours.END}')
    return trials, size, all_degrees, strategies, cost_mappings


def get_graph(graph_type, p, size):
    """
    Takes a string parameter indicating graph class and generation parameters, returns the requested graph if possible.
    :param graph_type: graph class requested (will lead to read-in from CSV or random generation as appropriate)
    :param p: generation parameter if appropriate (e.g. probability for ER graph, m for BA graph)
    :param size: number of vertices
    :return: requested graph
    """
    if isinstance(graph_type, nx.Graph):
        return graph_type, -1, [degree for node, degree in graph_type.degree()]
    elif isinstance(graph_type, str):
        if os.path.isfile(f'network-data-in/{graph_type}.csv'):
            g = read_graph_from_csv(graph_type, p)
            return g, -1, [degree for node, degree in g.degree()]
        else:
            seed = random.randint(0, 2 ** 32 - 1)
            if graph_type.lower() in ['erdos renyi', 'erdos-renyi', 'er']:
                g = nx.erdos_renyi_graph(size, p, seed=seed)
                return g, seed, [degree for node, degree in g.degree()]
            elif graph_type.lower() in ['barabasi-albert', 'barabasi albert', 'ba']:
                g = nx.barabasi_albert_graph(n=size, m=p, seed=seed)
                return g, seed, [degree for node, degree in g.degree()]
            elif graph_type.lower() in ['random geometric', 'rand geom', 'geometric random', 'geom rand']:
                g = nx.random_geometric_graph(n=size, radius=p, seed=seed)
                return g, seed, [degree for node, degree in g.degree()]
            elif ' ' in graph_type and graph_type.split()[0] in ['random'] and graph_type.split()[1][1:] == "-regular":
                if p * size % 2 == 0:
                    g = nx.random_regular_graph(d=p, n=size, seed=seed)
                    return g, seed, [degree for node, degree in g.degree()]
                else:
                    print(f'\ngenerating random {p}-regular with {size + 1} nodes rather than {size} \n'
                          f'  since num nodes * d must be even for random d-reg graph')
                    g = nx.random_regular_graph(d=p, n=(size + 1), seed=seed)
                    return g, seed, [degree for node, degree in g.degree()]
            else:
                raise Exception(f"\nSorry, I can't read this graph type: {graph_type}")
    raise Exception(f"\nSorry, I didn't recognise this graph type: {graph_type}, type: {type(graph_type)}")


def read_graph_from_csv(graph_type, p=-1):
    """
    Helper method to read-in graph data from a CSV file and produce a NetworkX graph.
    :param graph_type: which graph to read-in (from hard-coded data input directory).
    :param p: parameter to filter data by if required (e.g. time step, edge weight).
    :return: requested graph, read-in from a CSV file, as a NetworkX graph.
    """
    df = pd.read_csv(f'network-data-in/{graph_type}.csv', sep=None, engine='python')
    if len(df.columns) < 2:
        raise ValueError("CSV file must have at least two columns for source and target nodes")
    df.columns = df.columns.astype(str)  # Ensure column names are strings
    df = df.rename(columns={df.columns[0]: 'source', df.columns[1]: 'target'})
    # Check for edge weight and time columns
    if len(df.columns) > 2:
        df = df.rename(columns={df.columns[2]: 'weight'})
    if len(df.columns) > 3:
        df = df.rename(columns={df.columns[3]: 'time'})
    if p != -1:
        if 'time' in df.columns and p in df['time'].unique():
            df = df[df['time'] == p]
        elif 'weight' in df.columns and p in df['weight'].unique():
            df = df[df['weight'] == p]
        else:
            raise ValueError(f"Value {p} not found in 'time' or 'weight' columns of CSV file")
    if len(df.columns) > 2:
        g = nx.from_pandas_edgelist(df, 'source', 'target', edge_attr=df.columns[2:].tolist())
    else:
        g = nx.from_pandas_edgelist(df, 'source', 'target')
    return g


def cost_firefighter(graph, burning, protected, heuristic, cost_function_type, budget, num_rounds=50, info=False,
                     temporal=None):
    """
    Main simulation code to run the firefighter problem on a given graph with a given heuristic and cost function.
    :param graph: graph on which to run simulation.
    :param burning: set of burning vertices (initially, just contains the root).
    :param protected: set of protected vertices (initially empty)
    :param heuristic: which heuristic to use.
    :param cost_function_type: which cost function to use.
    :param budget: maximum sum of costs for vertices selected for defence in a given turn.
    :param num_rounds: maximum lifetime of the problem,
    :param info: whether to print extra information to console during the simulation.
    :param temporal: whether the graph is temporal (i.e. has time-stamped edges).
    :return: the number of vertices saved by the firefighter, the strategy used, and the cost mapping.
    """
    if temporal is None:
        temporal = any(
            attr in data for _, _, data in graph.edges(data=True) for attr in ['time', 'week', 'month', 'year'])
    if info:
        print("\n\n\nNEW RUN")
    total_nodes = len(graph.nodes())
    strategy, cost_mapping = [], []
    for i in range(num_rounds):
        open_nodes = graph.nodes() - protected - burning
        if len(open_nodes) == 0:
            break
        # choose something to protect
        cost_fn = CostFunction(cost_function_type, max_cost=budget)
        threat_dict = populate_threat_dict(graph, burning, protected)
        costs = cost_fn.cost(graph, threat_dict=threat_dict)
        cost_mapping.append(costs)
        if info:
            print("Threat dict", threat_dict)
            print("COST FUNCTION")
            print(cost_fn, cost_function_type)
            print(costs)

        total_cost = 0
        round_strategy = []
        while total_cost < budget:
            open_nodes = graph.nodes() - protected - burning
            if len(open_nodes) == 0:
                break
            node_to_protect = heuristic.choose(graph, protected, burning, costs)
            total_cost += costs[node_to_protect]
            if total_cost < budget:
                protected.add(node_to_protect)
                round_strategy.append(node_to_protect)
        strategy.append(round_strategy)
        neighbours = set()
        for v in burning:
            if temporal:  # need to filter edges on current time step
                current_edges = [(u, w) for u, w, t in graph.edges(v, data='time') if t == i]
                neighbours.update([w for u, w in current_edges])
            else:
                neighbours.update(set(graph.neighbors(v)))
        neighbours -= set(protected)
        burning.update(neighbours)

        if info:
            print("Burning nodes")
            print(burning)
            print("Protected nodes")
            print(protected)
        if (len(burning) + len(protected)) == total_nodes:
            break
    return len(graph.nodes()) - len(burning), strategy, cost_mapping


def write_to_csv(data, path, filename, header):
    """
    Helper method used to write simulation results to CSV files for later verification and analysis.
    :param data: the data to be written to file.
    :param path: where to store the file.
    :param filename: name of the file to create.
    :param header: columns for the CSV file.
    """
    file_path = f'{path}/{filename}'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
    file_exists = os.path.exists(file_path)
    file_empty = file_exists and os.path.getsize(file_path) == 0
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or file_empty:
            writer.writerow(header)
        for row in data:
            writer.writerow(row)


def write_degrees_to_csv(degrees, path):
    header = ['Degrees']
    write_to_csv(degrees, path, 'degrees.csv', header)


def write_strategies_to_csv(strategies, path):
    data = [list(key) + [strategy] for key, strategy_list in strategies.items() for strategy in strategy_list]
    header = ['Graph Type', 'Size', 'P', 'Seed', 'Root', 'Budget', 'Cost Type', 'Choice Type', 'Strategy']
    write_to_csv(data, path, 'strategies.csv', header)


def write_cost_mappings_to_csv(cost_mappings, path):
    data = [list(key) + [cost_mapping] for key, cost_mapping_list in cost_mappings.items() for cost_mapping in
            cost_mapping_list]
    header = ['Graph Type', 'Size', 'P', 'Seed', 'Root', 'Budget', 'Cost Type', 'Choice Type', 'Cost Mapping']
    write_to_csv(data, path, 'cost_mappings.csv', header)


def write_results_to_file(graph, path, results):
    graph_name = str(graph).split('/')[-1] if '/' in str(graph) else str(graph)
    filename = f'{graph_name.replace(" ", "_")}/results.csv'
    header = ["graph_type", "num_vertices", "parameter", "seed", "outbreak", "budget", "cost_function", "heuristic", "num_vertices_saved"]
    data = [list(r) + [results[r]] for r in results]
    write_to_csv(data, path, filename, header)


def main():
    budgets = [1, 2, 3, 4, 5]
    cost_functions = [CFn.UNIFORM,
                      CFn.UNIFORMLY_RANDOM,
                      CFn.STOCHASTIC_THREAT_LO,
                      CFn.STOCHASTIC_THREAT_HI,
                      CFn.HESITANCY_BINARY]

    heuristics = []
    heuristic_choices = [HeuristicChoices.RANDOM, HeuristicChoices.DEGREE, HeuristicChoices.THREAT,
                         HeuristicChoices.COST]
    for h1 in heuristic_choices:
        heuristics.append(Heuristic(h1, None))
        for h2 in heuristic_choices:
            if h1 != h2 and h1 != HeuristicChoices.RANDOM:
                could_add = Heuristic(h1, h2)
                if could_add not in heuristics:
                    heuristics.append(could_add)

    num_vertices = 50
    num_trials = 50
    outbreak = 'rand'  # can be an int vertex or 'rand' for random outbreak each time
    graph_types = {
        "mammalia-raccoon-proximity": [-1],
        "tnet-malawi-pilot": [-1],
        "reptilia-lizard-network-social": [-1],
        "random geometric": [0.1, 0.25, 0.5],
        "random n-regular": [1, 2, 3, 4],
        "barabasi-albert": [1, 2, 3, 4, 5],
        "erdos-renyi": [0.05, 0.1, 0.15, 0.2, 0.25]
    }

    start = time.time()
    exp_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S") # to avoid overwriting previous results

    for budget in budgets:
        print(f'\nbudget: {budget}')
        for cost_type in cost_functions:
            print(f'cost function: {cost_type}')
            path = f'output/{exp_id}/{cost_type}'
            os.makedirs(path, exist_ok=True)

            each_cost_start = time.time()

            for graph in graph_types.keys():
                print(f'graph: {graph}')
                each_graph_start = time.time()
                results = {}
                for each_param in graph_types[graph]:
                    print(f'input param.: {each_param}')
                    latest_results, actual_num_vertices, degrees, strategies, cost_mappings = run_experiments(
                        num_vertices, each_param,
                        num_trials,
                        graph, heuristics,
                        cost_type=cost_type,
                        budget=budget,
                        outbreak=outbreak)

                    results.update(latest_results)
                    each_graph_end = time.time() - each_graph_start

                    write_degrees_to_csv(degrees, f'{path}/{graph.replace(" ", "_")}')
                    write_strategies_to_csv(strategies, f'{path}/{graph.replace(" ", "_")}')
                    write_cost_mappings_to_csv(cost_mappings, f'{path}/{graph.replace(" ", "_")}')

                    all_costs = []
                    for cost_mapping_list in cost_mappings.values():
                        for cost_mapping in cost_mapping_list:
                            for costs in cost_mapping:
                                all_costs.extend(costs.values())

                    print(f'considered {graph} under {cost_type}, took {each_graph_end:.2f} secs.')
                    # print('now plotting...')
                    # plot_time = plot_helper(heuristics=heuristics, cost_type=cost_type, each_graph=graph,
                    #                         each_param=each_param, latest_results=latest_results,
                    #                         num_vertices=actual_num_vertices, budget=budget,
                    #                         costs=all_costs, degrees=degrees, location=path)
                    # print(f'(took {plot_time:.2f}s to plot)')
                # when we have considered all params, write results to a csv file
                write_results_to_file(graph, path, results)
            # finished considering a cost function, stop timer
            each_cost_end = time.time() - each_cost_start
            print(f'end of considering {cost_type}, took {each_cost_end:.2f} secs')

        time_taken = time.time() - start
        print(f'All done! time taken: {datetime.timedelta(seconds=time_taken)} ({time_taken} secs)')


# def write_results_to_file(graph, path, results):
#     graph_name = str(graph).split('/')[-1] if '/' in str(graph) else str(graph)
#     filename = f'{path}/{graph_name.replace(" ", "_")}/results.csv'
#     print(f'Writing to file {filename} ...')
#     file_write_start = time.time()
#     # Check if the file exists and is empty
#     file_exists = os.path.exists(filename)
#     file_empty = file_exists and os.path.getsize(filename) == 0
#     with open(filename, "a") as f:
#         if not file_exists or file_empty:
#             f.write(
#                 "graph_type,num_vertices,parameter,seed,outbreak,budget,cost_function,heuristic,num_vertices_saved\n")
#
#         for r in results:
#             formatted_r = [str(x).replace(' ', '_') if isinstance(x, str) else str(x) for x in r]
#             f.write(','.join(formatted_r) + ',' + ','.join(map(str, results[r])) + '\n')
#     print(f'Finished writing to {filename} in {time.time() - file_write_start:.2f} secs')


if __name__ == main():
    main()


class Colours:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
