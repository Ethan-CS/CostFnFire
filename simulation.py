import csv
import datetime
import os
import random
import time
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Dict, List

import networkx as nx
import pandas as pd

from costs_and_heuristics import CostFunction, Heuristic, populate_threat_dict, CFn, HeuristicChoices


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


# main simulation code used to run Cost function firefighter simulations on given input graph(s)

def _run_single_trial(args) -> Tuple[List[int], Dict, Dict, Dict]:
    graph_type, p, size, heuristics, cost_type, budget, outbreak = args
    g, seed, degrees = get_graph(graph_type, p, size)
    actual_size = g.order()
    results_local, strategies_local, cost_mappings_local = {}, {}, {}
    if outbreak is None:
        root = list(g.nodes)[0]
    elif outbreak in g.nodes:
        root = outbreak
    else:
        root = random.choice(list(g.nodes))
    for choice_type in heuristics:
        input_params = (graph_type, actual_size, p, seed, root, budget, cost_type, choice_type)
        results_local.setdefault(input_params, [])
        strategies_local.setdefault(input_params, [])
        cost_mappings_local.setdefault(input_params, [])
        saved, strategy, cost_mapping = cost_firefighter(
            g, {root}, set([]), choice_type, cost_type, num_rounds=int(actual_size / 2), budget=budget, info=False
        )
        results_local[input_params].append(saved)
        strategies_local[input_params].append(strategy)
        cost_mappings_local[input_params].append(cost_mapping)
    return degrees, results_local, strategies_local, cost_mappings_local


def run_experiments(size, p, num_trials, graph_type, heuristics, cost_type, budget, outbreak, progress=True, workers=1):
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
    :param workers: number of parallel worker processes for trials.
    :return: simulation results in a dictionary, with format:
    `(graph_type, size, p, seed, outbreaks, budget, cost_type, choice_type): num_saved_vertices`
    """
    trials, strategies, cost_mappings = {}, {}, {}
    all_degrees: List[List[int]] = []
    actual_size = size

    if progress:
        print(f'{Colours.GREEN}-' * num_trials, f'100% ({num_trials} trials)')
    if workers and workers > 1 and num_trials > 1:
        # Parallel execution across trials
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_run_single_trial, (graph_type, p, size, heuristics, cost_type, budget, outbreak)) for
                       _ in range(num_trials)]
            for fut in as_completed(futures):
                if progress:
                    print("-", end="", flush=True)
                degrees, res_l, strat_l, costs_l = fut.result()
                if not all_degrees:
                    all_degrees.append(degrees)
                for k, v in res_l.items():
                    trials.setdefault(k, []).extend(v)
                for k, v in strat_l.items():
                    strategies.setdefault(k, []).extend(v)
                for k, v in costs_l.items():
                    cost_mappings.setdefault(k, []).extend(v)
        # derive actual size from any key
        if trials:
            any_key = next(iter(trials.keys()))
            actual_size = any_key[1]
    else:
        # Serial execution
        for _ in range(num_trials):
            if progress:
                print("-", end="", flush=True)
            g, seed, degrees = get_graph(graph_type, p, size)
            if not all_degrees:
                all_degrees.append(degrees)
            actual_size = g.order()
            for choice_type in heuristics:
                if outbreak is None:
                    root = list(g.nodes)[0]
                elif outbreak in g.nodes:
                    root = outbreak
                else:
                    root = random.choice(list(g.nodes))
                input_params = (graph_type, actual_size, p, seed, root, budget, cost_type, choice_type)
                trials.setdefault(input_params, [])
                strategies.setdefault(input_params, [])
                cost_mappings.setdefault(input_params, [])
                saved, strategy, cost_mapping = cost_firefighter(
                    g, {root}, set([]), choice_type, cost_type, num_rounds=int(actual_size / 2), budget=budget,
                    info=False
                )
                trials[input_params].append(saved)
                strategies[input_params].append(strategy)
                cost_mappings[input_params].append(cost_mapping)
    if progress:
        print(f" 100% - {Colours.UNDERLINE}complete")
        print(f'{Colours.END}')
    return trials, actual_size, all_degrees, strategies, cost_mappings


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
            # Ensure simple undirected for downstream logic
            if not isinstance(g, nx.Graph) or g.is_directed():
                g = nx.Graph(g)
            return g, -1, [degree for node, degree in g.degree()]
        else:
            seed = random.randint(0, 2 ** 32 - 1)
            gt = graph_type.lower()
            if gt in ['erdos renyi', 'erdos-renyi', 'er']:
                g = nx.erdos_renyi_graph(size, p, seed=seed)
                return g, seed, [degree for node, degree in g.degree()]
            elif gt in ['barabasi-albert', 'barabasi albert', 'ba']:
                m = max(1, min(int(p), size - 1))
                g = nx.barabasi_albert_graph(n=size, m=m, seed=seed)
                return g, seed, [degree for node, degree in g.degree()]
            elif gt in ['random geometric', 'rand geom', 'geometric random', 'geom rand']:
                g = nx.random_geometric_graph(n=size, radius=float(p), seed=seed)
                return g, seed, [degree for node, degree in g.degree()]
            elif ' ' in graph_type and graph_type.split()[0] in ['random'] and graph_type.split()[1][1:] == "-regular":
                d = int(p)
                if d >= size:
                    d = size - 1
                if (d * size) % 2 == 1:
                    # Adjust size by +1 to satisfy existence condition
                    print(f'\nrandom {d}-regular requires d*n even; using {size + 1} nodes instead of {size}')
                    g = nx.random_regular_graph(d=d, n=(size + 1), seed=seed)
                else:
                    g = nx.random_regular_graph(d=d, n=size, seed=seed)
                return g, seed, [degree for node, degree in g.degree()]
            elif gt in ['watts-strogatz', 'small-world', 'ws']:
                k = int(p)
                if k % 2 == 1:
                    k += 1  # ensure even as required by WS
                k = min(max(2, k), size - 1)
                g = nx.watts_strogatz_graph(n=size, k=k, p=0.1, seed=seed)
                return g, seed, [degree for node, degree in g.degree()]
            elif gt in ['powerlaw-cluster', 'plc']:
                m = max(1, min(int(p), size - 1))
                g = nx.powerlaw_cluster_graph(n=size, m=m, p=0.1, seed=seed)
                return g, seed, [degree for node, degree in g.degree()]
            elif gt in ['scale-free', 'sf']:
                # Convert to simple undirected graph to avoid MultiDiGraph pitfalls
                g = nx.Graph(nx.scale_free_graph(n=size, seed=seed))
                return g, seed, [degree for node, degree in g.degree()]
            elif gt in ['connected-caveman', 'caveman']:
                clique_size = max(1, int(p))
                num_cliques = max(1, size // clique_size)
                # Use correct parameter order: (number_of_cliques, size_of_clique)
                g = nx.connected_caveman_graph(num_cliques, clique_size)
                return g, seed, [degree for node, degree in g.degree()]
            elif gt in ['random-lobster', 'lobster']:
                g = nx.random_lobster(n=size, p1=0.5, p2=0.5, seed=seed)
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
    if not isinstance(g, nx.Graph) or g.is_directed():
        g = nx.Graph(g)
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
            node_cost = costs[node_to_protect]
            # Prevent infinite loops caused by pathological zero-cost values (shouldn't happen, but be safe)
            if node_cost <= 0:
                break
            total_cost += node_cost
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
    header = ["graph_type", "num_vertices", "parameter", "seed", "outbreak", "budget", "cost_function", "heuristic",
              "num_vertices_saved"]
    data = [list(r) + [results[r]] for r in results]
    write_to_csv(data, path, filename, header)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run firefighter simulations with selectable graphs and parameters")
    parser.add_argument("--graph", type=str, default=None,
                        help="Graph type to run (e.g., 'erdos-renyi', 'barabasi-albert', 'random geometric', 'scale-free')")
    parser.add_argument("--params", type=str, default=None,
                        help="Comma-separated list of parameter values for the chosen graph (e.g., '0.05,0.1' or '2,3,4')")
    parser.add_argument("--size", type=int, default=100, help="Number of vertices (default: 100)")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials per configuration (default: 50)")
    parser.add_argument("--budgets", type=str, default="1,2,3,4,5", help="Comma-separated budgets (default: 1,2,3,4,5)")
    parser.add_argument("--costs", type=str, default=None,
                        help="Comma-separated cost functions by name (e.g., 'Uniform,Uniformly Random,Stochastic-Threat (high)'); default is all")
    parser.add_argument("--heuristics", type=str, default=None,
                        help="Comma-separated heuristics, optionally with tie-breakers using '/', e.g., 'Random,Degree,Threat/Cost'; default is all")
    parser.add_argument("--outbreak", type=str, default="rand", help="Outbreak node id or 'rand' (default)")
    parser.add_argument("--output-dir", type=str, default="output", help="Base output directory (default: output)")
    parser.add_argument("--exp-id", type=str, default=None, help="Experiment id; default is timestamp")
    parser.add_argument("--progress", action="store_true", help="Print per-trial progress dashes")
    parser.add_argument("--save-details", action="store_true", help="Also write strategies and cost mappings (large)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of parallel worker processes for trials (default: 1)")
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])

    # Defaults
    budgets = [1, 2, 3, 4, 5]
    cost_functions = [CFn.UNIFORM, CFn.UNIFORMLY_RANDOM, CFn.STOCHASTIC_THREAT_LO, CFn.STOCHASTIC_THREAT_HI,
                      CFn.HESITANCY_BINARY]

    # Derive heuristics list
    heuristics: list[Heuristic] = []
    if args.heuristics:
        for h in [x.strip() for x in args.heuristics.split(',') if x.strip()]:
            heuristics.append(Heuristic.from_string(h))
    else:
        heuristic_choices = [HeuristicChoices.RANDOM, HeuristicChoices.DEGREE, HeuristicChoices.THREAT,
                             HeuristicChoices.COST]
        for h1 in heuristic_choices:
            heuristics.append(Heuristic(h1, None))
            for h2 in heuristic_choices:
                if h1 != h2 and h1 != HeuristicChoices.RANDOM:
                    could_add = Heuristic(h1, h2)
                    if could_add not in heuristics:
                        heuristics.append(could_add)

    num_vertices = args.size
    num_trials = args.trials
    outbreak = None if args.outbreak.lower() == 'first' else args.outbreak  # 'rand' or node id

    # Cost functions filter
    if args.costs:
        name_map = {str(c): c for c in
                    [CFn.UNIFORM, CFn.UNIFORMLY_RANDOM, CFn.STOCHASTIC_THREAT, CFn.STOCHASTIC_THREAT_LO,
                     CFn.STOCHASTIC_THREAT_MID, CFn.STOCHASTIC_THREAT_HI, CFn.HESITANCY_BINARY]}
        # Accept case-insensitive keys of the names
        inv = {k.lower(): v for k, v in name_map.items()}
        parsed = []
        for name in [x.strip() for x in args.costs.split(',') if x.strip()]:
            key = name.lower()
            if key in inv:
                parsed.append(inv[key])
            else:
                raise ValueError(f"Unknown cost function '{name}'. Valid options: {', '.join(name_map.keys())}")
        if parsed:
            cost_functions = parsed

    # Budgets
    budgets = [int(x) for x in args.budgets.split(',') if x.strip()]

    # Graph selection
    graph_types = {}
    if args.graph:
        # user-specified single graph
        if args.params:
            params = []
            for token in args.params.split(','):
                token = token.strip()
                if token == '':
                    continue
                try:
                    params.append(int(token))
                except ValueError:
                    params.append(float(token))
        else:
            # default params per graph
            defaults = {
                'erdos-renyi': [0.05, 0.1, 0.15, 0.2, 0.25],
                'barabasi-albert': [1, 2, 3, 5, 8, 10],
                'watts-strogatz': [4, 6, 8, 10],
                'powerlaw-cluster': [2, 3, 4, 5],
                'scale-free': [-1],
                'random geometric': [0.05, 0.1, 0.15, 0.2, 0.25],
                'random n-regular': [2, 3, 4, 6, 8],
                'connected-caveman': [5, 10, 15, 20],
                'random-lobster': [-1],
            }
            params = defaults.get(args.graph, [-1])
        graph_types = {args.graph: params}
    else:
        # Original default sweep
        graph_types = {
            # "mammalia-raccoon-proximity": [-1],
            # "tnet-malawi-pilot": [-1],
            # "reptilia-lizard-network-social": [-1],
            "erdos-renyi": [0.05, 0.1, 0.15, 0.2, 0.25],
            "barabasi-albert": [1, 2, 3, 5, 8, 10],
            "watts-strogatz": [4, 6, 8, 10],  # k parameter (avg degree)
            "powerlaw-cluster": [2, 3, 4, 5],  # m parameter
            "scale-free": [-1],  # no parameter needed
            "random geometric": [0.05, 0.1, 0.15, 0.2, 0.25],
            "random n-regular": [2, 3, 4, 6, 8],
            "connected-caveman": [5, 10, 15, 20],  # clique sizes
            "random-lobster": [-1],
        }

    start = time.time()
    exp_id = args.exp_id or datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # to avoid overwriting previous results

    for budget in budgets:
        print(f"\nbudget: {budget}")
        for cost_type in cost_functions:
            print(f"cost function: {cost_type}")
            path = f"{args.output_dir}/{exp_id}/{cost_type}"
            os.makedirs(path, exist_ok=True)

            each_cost_start = time.time()

            for graph in graph_types.keys():
                print(f"graph: {graph}")
                results = {}
                for each_param in graph_types[graph]:
                    each_graph_start = time.time()
                    print(f"input param.: {each_param}")
                    latest_results, actual_num_vertices, degrees, strategies, cost_mappings = run_experiments(
                        num_vertices, each_param, num_trials, graph, heuristics,
                        cost_type=cost_type, budget=budget, outbreak=outbreak, progress=args.progress,
                        workers=args.jobs)

                    results.update(latest_results)
                    each_graph_end = time.time() - each_graph_start

                    # Always write degrees; write heavy details only on request
                    write_degrees_to_csv(degrees, f'{path}/{graph.replace(" ", "_")}')
                    if args.save_details:
                        write_strategies_to_csv(strategies, f'{path}/{graph.replace(" ", "_")}')
                        write_cost_mappings_to_csv(cost_mappings, f'{path}/{graph.replace(" ", "_")}')

                    print(f"considered {graph} under {cost_type}, took {each_graph_end:.2f} secs.")
                # when we have considered all params, write results to a csv file
                write_results_to_file(graph, path, results)
            # finished considering a cost function, stop timer
            each_cost_end = time.time() - each_cost_start
            print(f"end of considering {cost_type}, took {each_cost_end:.2f} secs")

        time_taken = time.time() - start
        print(f"All done! time taken: {datetime.timedelta(seconds=time_taken)} ({time_taken} secs)")


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


if __name__ == "__main__":
    main()
