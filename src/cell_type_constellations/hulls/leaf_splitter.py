import numpy as np
import scipy.spatial

class HullGraph(object):

    def __init__(self):
        self._nodes = set()
        self._edges = dict()

    def add_node(self, node):
        assert node not in self._nodes
        assert node not in self._edges
        self._nodes.add(node)
        self._edges[node] = set()

    def add_bidirectional_edge(self, node0, node1):
        if node0 not in self._nodes:
            self.add_node(node0)
        if node1 not in self._nodes:
            self.add_node(node1)
        self._edges[node0].add(node1)
        self._edges[node1].add(node0)

    def clip_bidirectional_edge(self, node0, node1):
        if node1 in self._edges[node0]:
            self._edges[node0].remove(node1)
        if node0 in self._edges[node1]:
            self._edges[node1].remove(node0)

    def partition_graph(self):
        node_sets = []
        already_visited = set()
        while True:
            (subgraph_nodes,
             already_visited)= get_subgraph(
                input_node_set=self._nodes,
                edge_lookup=self._edges,
                already_visited=already_visited,
                starting_node=None)
            if len(subgraph_nodes) > 0:
                node_sets.append(subgraph_nodes)
            else:
                break
        return node_sets


def get_subgraph(
        input_node_set,
        edge_lookup,
        already_visited,
        starting_node):
    output_node_set = set()
    if starting_node is None:
        for node in input_node_set:
            if node not in already_visited:
                starting_node = node
                break

    node_queue = set()
    next_node = starting_node
    while next_node is not None:
        output_node_set.add(next_node)
        already_visited.add(next_node)
        for candidate in edge_lookup[next_node]:
            if candidate not in already_visited:
                node_queue.add(candidate)

        if len(node_queue) > 0:
            next_node = node_queue.pop()
        else:
            next_node = None

    return (output_node_set, already_visited)


def subdivide_points(point_array, k_nn=20, n_sig=2):
    if point_array.shape[0] < 10:
        return [range(point_array.shape[0]),]

    nn_graph = create_hull_graph(
        point_array=point_array,
        k_nn=k_nn,
        n_sig=n_sig
    )

    sub_populations = nn_graph.partition_graph()
    return sub_populations


def create_hull_graph(point_array, k_nn, n_sig):

    if k_nn > point_array.shape[0]-1:
        k_nn = point_array.shape[0]-1
    
    kd_tree = scipy.spatial.cKDTree(point_array)
    raw_nn = kd_tree.query(point_array, k=k_nn+1)
    distance = raw_nn[0][:, 1:]
    nn_idx = raw_nn[1][:, 1:]
    del raw_nn

    global_99 = np.quantile(distance, 0.99)
    
    nn_graph = HullGraph()
    for idx in range(point_array.shape[0]):
        nn_graph.add_node(idx)

    #print(f'point_array shape {point_array.shape}')
    #print(f'distance shape {distance.shape}')
    #print(f'nn_idx shape {nn_idx.shape}')
    
    edge_candidates = dict()
    for i0 in range(point_array.shape[0]):

        chosen_idx = np.concatenate([[i0], nn_idx[i0, :]])

        sub_idx = nn_idx[chosen_idx, :]
        sub_distances = np.concatenate(
            [distance[chosen_idx, :].flatten(),
             distance[sub_idx, :].flatten()]
        )

        q25, q50, q75 = np.quantile(sub_distances, (0.25, 0.5, 0.75))
        mu = q50
        std = (q75-q25)/1.35
        max_distance = min(global_99, mu+n_sig*std)
        #max_distance = mu+n_sig*std
        edge_candidates[i0] = set(nn_idx[i0, :][np.where(distance[i0, :] <= max_distance)])

    for i0 in edge_candidates:
        for i1 in edge_candidates[i0]:
            if i0 in edge_candidates[i1]:
                nn_graph.add_bidirectional_edge(i0, i1)

    return nn_graph


def iteratively_subdivide_points(point_array, k_nn=20, n_sig=2):
    base_idx = np.array(range(point_array.shape[0]))
    current_subdivision = [base_idx,]

    while True:
        next_subdivision = []
        for parent_subset in current_subdivision:
            pts = point_array[
                parent_subset, :
            ]
            this_subdivision = subdivide_points(point_array=pts, k_nn=k_nn, n_sig=n_sig)
            for child_subset in this_subdivision:
                child_idx = np.sort(np.array(list(child_subset)))
                child_idx = parent_subset[child_idx]
                next_subdivision.append(child_idx)
        #print(len(next_subdivision), len(current_subdivision))
        if len(next_subdivision) == len(current_subdivision):
            break
        current_subdivision = next_subdivision

    return current_subdivision
