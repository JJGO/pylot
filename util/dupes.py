import itertools
from typing import List, Dict
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import pdist


def undirected_connected_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    components = []
    seen = set()
    for n in graph:
        if n not in seen:
            components.append([])
            seen.add(n)
            stack = [n]
            while stack:
                n = stack.pop()
                components[-1].append(n)
                for m in graph[n]:
                    if m not in seen:
                        seen.add(m)
                        stack.append(m)
    return components


# Function to find sets of duplicate rows
def array_duplicate_rowsets(arrays: np.ndarray, metric="cityblock", names=None):
    # compute distances and threshold
    indices = np.arange(len(arrays))
    pair_dist = np.isclose(pdist(arrays, metric=metric), 0)
    # build graph
    print(pair_dist)
    graph = defaultdict(list)
    for (i, j), d in zip(itertools.combinations(indices, 2), pair_dist):
        if d:
            graph[i].append(j)
            graph[j].append(i)
    # connected components
    dupe_sets = undirected_connected_components(graph)
    if names:
        dupe_sets = [[names[i] for i in dupe_set] for dupe_set in dupe_sets]
    return dupe_sets
