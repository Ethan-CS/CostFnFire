import datetime
import time

import networkx as nx

import junk
from choices import Heur, CFn
from costs_and_heuristics import CostFunction, Heuristic, populate_threat_dict
from plotting import plot_helper


def run_experiments(size, p, num_trials, graph_type, heuristics, cost_type, budget=5, progress=True):
    trials = {}
    # for graph in graph_types:
    choice_functions = [Heuristic(name[0], name[1]) for name in heuristics]

    if progress:
        print(f'{junk.Colours.GREEN}-' * num_trials, f'100% ({num_trials} trials)')
    for i in range(num_trials):
        if progress:
            print("-", end="")
        g = get_graph(graph_type, p, size)

        for choice_type in choice_functions:
            saved = cost_firefighter(g, {0}, set([]), choice_type, cost_type,
                                     num_rounds=int(g.number_of_nodes() / 2), budget=budget, INFO=False)
            if (graph_type, size, p, cost_type, (choice_type.which_heuristic, choice_type.tie_break)) not in trials:
                trials[(graph_type, size, p, cost_type, (choice_type.which_heuristic, choice_type.tie_break))] = []
            trials[(graph_type, size, p, cost_type, (choice_type.which_heuristic, choice_type.tie_break))].append(saved)
    if progress:
        print(f" 100% - {junk.Colours.UNDERLINE}complete")
        print(f'{junk.Colours.END}')
    return trials


def get_graph(graph_type, p, size):
    if graph_type.lower() in ['erdos renyi', 'er']:
        g = nx.barabasi_albert_graph(size, p)
    elif graph_type.lower() in ['barabasi-albert', 'barabasi albert', 'ba']:
        g = nx.barabasi_albert_graph(m=p, n=size)
    elif graph_type.lower() in ['random geometric', 'rand geom', 'geometric random', 'geom rand']:
        g = nx.random_geometric_graph(radius=p, n=size)
    elif ' ' in graph_type and graph_type.split()[0] in ['random'] and graph_type.split()[1][1:] == "-regular":
        if p * size % 2 == 0:
            g = nx.random_regular_graph(d=p, n=size)  # we take p to be n in 'random n-regular'
        else:
            print(f'\ngenerating random {p}-regular with {size + 1} nodes rather than {size} \n'
                  f'  since num nodes * d must be even for random d-reg graph')
            g = nx.random_regular_graph(d=p, n=(size + 1))
    else:
        raise Exception(f"\nSorry, I didn't recognise this graph type: {graph_type}")
    return g


def cost_firefighter(graph, burning, protected, heuristic, cost_function_type, num_rounds=50, INFO=True,
                     budget=5):
    if INFO:
        print("\n\n\nNEW RUN")
    total_nodes = len(graph.nodes())

    last_burning = 0
    for _ in range(num_rounds):
        open = graph.nodes() - protected - burning
        if len(open) == 0:
            break
        # choose something to protect
        cost_fn = CostFunction(cost_function_type)
        threat_dict = populate_threat_dict(graph, burning, protected)
        costs = cost_fn.cost(graph, threat_dict=threat_dict)
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
        last_burning = len(burning)
    return len(graph.nodes()) - len(burning)


def main():
    cost_functions = [CFn.STOCHASTIC_THREAT,
                      CFn.HESITANCY_BINARY,
                      CFn.UNIFORMLY_RANDOM]

    heuristics = [
        (Heur.DEGREE, None),
        (Heur.THREAT, None),
        (Heur.COST, None),
        (Heur.DEGREE, Heur.THREAT),
        (Heur.DEGREE, Heur.COST),
        (Heur.THREAT, Heur.COST),
        (Heur.THREAT, Heur.DEGREE),
        (Heur.COST, Heur.DEGREE),
        (Heur.COST, Heur.THREAT)]

    num_vertices = 25
    num_trials = 25
    budget = int(num_vertices / 10)  # TODO make this e.g. mean/median cost for each instance and box plot?

    graph_types = {"random geometric": [0.15],
                   "random n-regular": [4],
                   "Barabasi-Albert": [int(num_vertices / 16)]}

    start = time.time()
    for cost_type in cost_functions:
        each_cost_start = time.time()

        for each_graph in graph_types.keys():
            each_graph_start = time.time()
            results = {}
            for each_param in graph_types[each_graph]:
                print(f'Cost fn.: {cost_type}, graph type: {each_graph}, input param.: {each_param}')
                latest_results = run_experiments(num_vertices, each_param, num_trials, each_graph, heuristics,
                                                 cost_type=cost_type, budget=budget)
                results.update(latest_results)
                each_graph_end = time.time() - each_graph_start
                print(
                    f'end of considering {each_graph} with {num_vertices} vx, param {each_param}, under {cost_type}, '
                    f'took {each_graph_end} secs - now plotting...'
                )
                plot_time = plot_helper(heuristics, cost_type, each_graph, each_param, latest_results, num_vertices,
                                        results, budget, violin_too=True)
                print(f'(took {plot_time}s to plot)')
        each_cost_end = time.time() - each_cost_start
        print(f'end of considering {cost_type}, took {each_cost_end} secs')
    print('All done!')
    time_taken = time.time() - start
    print(f'TOTAL time taken: {datetime.timedelta(seconds=time_taken)} ({time_taken} secs)')


if __name__ == main():
    main()

# add a periodic cost function?
# a heuristic that is in-proportion stochastically to cost?
# OR even better a cost that is stochastically related to threat
# Those are done

# Better visualisations
