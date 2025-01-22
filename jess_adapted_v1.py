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
    all_degrees = []
    if progress:
        print(f'{junk.Colours.GREEN}-' * num_trials, f'100% ({num_trials} trials)')

    for i in range(num_trials):
        if progress:
            print("-", end="")

        g, seed, degrees = get_graph(graph_type, p, size)
        all_degrees.extend(degrees)

        if g.order() != size:  # e.g. read-in from csv, return actual size for plotting etc.
            size = g.order()
        for choice_type in heuristics:
            if outbreak is None:
                root = list(g.nodes)[0]
            elif outbreak in g.nodes:
                root = outbreak
            else:  # e.g. root is 'rand,' indicating choose a random vertex each time
                root = random.choice(list(g.nodes))

            input_params = (graph_type, size, p, seed, root, cost_type, choice_type)
            trials.setdefault(input_params, [])

            saved = cost_firefighter(g, {root}, set([]), choice_type, cost_type, num_rounds=int(g.number_of_nodes() / 2), budget=budget, INFO=False)

            trials[input_params].append(saved[0])
            all_costs.extend(saved[1])
    if progress:
        print(f" 100% - {junk.Colours.UNDERLINE}complete")
        print(f'{junk.Colours.END}')
    return trials, size, all_costs, all_degrees


def get_graph(graph_type, p, size):
    if isinstance(graph_type, nx.Graph):
        return graph_type, -1, [degree for node, degree in graph_type.degree()]
    elif isinstance(graph_type, str):
        if os.path.isfile(f'network-data-in/{graph_type}.csv'):
            try:
                df = pd.read_csv(f'network-data-in/{graph_type}.csv', sep=None, engine='python')
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
                    g = nx.from_pandas_edgelist(df, 'source', 'target', edge_attr=True)
                    return g, -1, [degree for node, degree in g.degree()]
                else:
                    g = nx.from_pandas_edgelist(df)
                    return g, -1, [degree for node, degree in g.degree()]
            except Exception as e:
                raise Exception(f"Error reading CSV file: {e}")

        seed = random.randint(0, 2 ** 32 - 1)
        if graph_type.lower() in ['erdos renyi', 'er']:
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
            raise Exception(f"\nSorry, I didn't recognise this graph type: {graph_type}, type: {type(graph_type)}")
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

    num_vertices = 20
    num_trials = 50
    budget = int(num_vertices / 8)  # TODO vary this sensibly
    outbreak = 'rand'  # can be an int ot 'rand' for random outbreak each time
    graph_types = {
        "mammalia-raccoon-proximity": [0],
        "tnet_malawi_pilot": [0],
        # "random geometric": [0.1, 0.5, 0.9],
        # "random n-regular": [2, 3, 4],
        # "Barabasi-Albert": [int(num_vertices / 16), int(num_vertices / 8)],
        "reptilia-tortoise-network-fi": [2005]
        }


    # TODO: more graphs, in particular varying connection topology (some more like preferential attachment, some more
    #  regular - in the latter, choosing by degree rubbish, in former, possibly better, cost will usually vary
    #  by cost distribution, part of the story to explain)

    # TODO: some experiments that mirror theoretical sections, e.g. trees, sea fans

    start = time.time()
    exp_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    for cost_type in cost_functions:
        path = f'output/{exp_id}/{cost_type}'
        os.makedirs(path, exist_ok=True)

        each_cost_start = time.time()

        for graph in graph_types.keys():
            print(f' ***** {graph} *****')
            each_graph_start = time.time()
            results = {}
            for each_param in graph_types[graph]:
                print(f'  cost fn.: {cost_type}, graph type: {graph}, input param.: {each_param}')
                latest_results, actual_num_vertices, costs, degrees = run_experiments(num_vertices, each_param, num_trials,
                                                                             graph, heuristics, cost_type=cost_type,
                                                                             budget=budget, outbreak=outbreak, )
                # latest results format:
                # (graph_type, size, p, seed, outbreaks, cost_type, choice_type): num_saved_vertices
                results.update(latest_results)

                each_graph_end = time.time() - each_graph_start

                print(
                    f'end of considering {graph} under {cost_type}, took {each_graph_end} secs - now plotting...')
                plot_time = plot_helper(heuristics=heuristics, cost_type=cost_type, each_graph=graph,
                                        each_param=each_param, latest_results=latest_results,
                                        num_vertices=actual_num_vertices, results=results, budget=budget, costs=costs,
                                        degrees=degrees, violin_too=True, location=path)
                print(f'(took {plot_time}s to plot)')

            graph_name = str(graph).split('/')[-1] if '/' in str(graph) else str(graph)
            filename = f'{path}/{graph_name.replace(" ", "_")}/results.csv'
            print(f'Writing to file {filename} ...')
            # Check if the file exists and is empty
            file_exists = os.path.exists(filename)
            file_empty = file_exists and os.path.getsize(filename) == 0

            with open(filename, "a") as f:
                if not file_exists or file_empty:
                    f.write(
                        "graph_type,num_vertices,parameter,seed,outbreak,cost_function,heuristic,num_vertices_saved\n")

                for r in results:
                    formatted_r = [str(x).replace(' ', '_') if isinstance(x, str) else str(x) for x in r]
                    f.write(','.join(formatted_r) + ',' + ','.join(map(str, results[r])) + '\n')

            print(f'Finished writing to {filename}')

        each_cost_end = time.time() - each_cost_start
        print(f'end of considering {cost_type}, took {each_cost_end} secs')
    print('All done!')
    time_taken = time.time() - start
    print(f'TOTAL time taken: {datetime.timedelta(seconds=time_taken)} ({time_taken} secs)')


if __name__ == main():
    main()
