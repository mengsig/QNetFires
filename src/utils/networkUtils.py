import networkx as nx
import numpy as np

def create_network(edgelist_path, sparse_array = False):
    if type(edgelist_path) is not str:
        edgelist = edgelist_path
    else:
        edgelist = np.loadtxt(edgelist_path)
    G = nx.DiGraph()
    for u, v, w in edgelist:
        G.add_edge(int(u), int(v), weight=np.log(float(w)+1)) 
    if sparse_array:
        GAdj = nx.to_scipy_sparse_array(G)
        return GAdj
    else:
        return G

def save_fuel_breaks(data, plot_degreec, basename, intervals, centrality):
    import os
    import sys
    script_dir   = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    sys.path.insert(0, project_root)
    from src.utils.plottingUtils import save_matrix_as_heatmap
    plot_degree = plot_degreec.copy()
    vmin = plot_degree.min()
    vmax = plot_degree.max()
    for cutoff in intervals:
        fuel_breaks = data > np.percentile(data, 100 - cutoff)
        plot_degree = plot_degreec.copy()
        plot_degree[fuel_breaks] = np.inf
        try:
            np.savetxt(f"{basename}_{cutoff}.txt", fuel_breaks)
            print(f"[GENERATING-FUEL-BREAKS-{centrality}:]: Saved file: {basename}_{cutoff}.txt")

            domirankConfig = {
                    "matrix"  : plot_degree,
                    "colors"  : "hot",
                    "units"   : "m/min",
                    "title"   : f"{centrality}",
                    "filename": f"{basename}_{cutoff}.png",
                    "vmin"    : vmin,
                    "vmax"    : vmax,
                    }
            save_matrix_as_heatmap(**domirankConfig)
        except:
            raise ValueError("Problem saving file")
    #plot original
    config = {
            "matrix"  : plot_degreec,
            "colors"  : "hot",
            "units"   : "m/min",
            "title"   : "adjacency",
            "filename": f"{basename}_adjacency.png",
            "vmin"    : plot_degreec.min(),
            "vmax"    : plot_degreec.max(),
            "norm"    : True
            }
    save_matrix_as_heatmap(**config)





#building network stuff used in src/scripts/create_adjacency.py
def build_edgelist_from_spread_rates(spread_rate_mean, x, y):
    """
    Constructs an adjacency list with self‐loops on boundary nodes
    to compensate for missing links.

    Parameters:
        spread_rate_mean (np.ndarray): shape (4 or 8, y, x), spread rates per direction:
            If 4 layers: 0=N, 1=E, 2=S, 3=W
            If 8 layers: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
        x (int): width of the grid.
        y (int): height of the grid.

    Returns:
        adjacency (list of tuples): (from_node, to_node, weight)
    """
    adjacency = []

    # 8‐neighborhood offsets (we’ll just ignore the diagonals if only 4 passed)
    directions = {
        0:  (0, -1),   # N
        1:  (1, -1),   # NE
        2:  (1, 0),    # E
        3:  (1, 1),    # SE
        4:  (0, 1),    # S
        5:  (-1, 1),   # SW
        6:  (-1, 0),   # W
        7:  (-1, -1),  # NW
    }

    # detect whether only 4 cardinal layers were supplied
    is_cardinal_only = (spread_rate_mean.shape[0] == 4)
    expected_links = 4 if is_cardinal_only else 8

    for j in range(y):
        for i in range(x):
            from_node = j * x + i
            # collect this node’s link‐weights
            neighbor_weights = []

            for d, (dx, dy) in directions.items():
                # skip diagonals if only 4 directions are available
                if is_cardinal_only and d not in (0,2,4,6):
                    continue

                ni, nj = i + dx, j + dy
                if not (0 <= ni < x and 0 <= nj < y):
                    continue

                # compute the weight for this direction
                if not is_cardinal_only:
                    weight = spread_rate_mean[d, j, i]
                else:
                    # map 4‐layer indices to [N,E,S,W]
                    if d in (0,2,4,6):
                        card_map = {0:0, 2:1, 4:2, 6:3}
                        weight = spread_rate_mean[card_map[d], j, i]
                    else:
                        # unreachable
                        continue

                to_node = nj * x + ni
                adjacency.append((from_node, to_node, weight))
                neighbor_weights.append(weight)

            # now add a self‐loop to make up for any missing links
            n_existing = len(neighbor_weights)
            missing = expected_links - n_existing
            if missing > 0 and n_existing > 0:
                mean_w = sum(neighbor_weights) / n_existing
                self_weight = mean_w * missing
                adjacency.append((from_node, from_node, self_weight))

    return adjacency
