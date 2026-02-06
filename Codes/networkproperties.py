import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from numba import njit
import pandas as pd
import math
import os
import time





def prepare_network_data(G):
    edge_vector = np.array(list(G.edges()))
    mirrored_edges = np.flip(edge_vector, axis=1)
    extended_edge_vector = np.vstack([edge_vector, mirrored_edges])
    extended_edge_vector = extended_edge_vector[extended_edge_vector[:, 0].argsort()]

    degrees = np.zeros(len(G), dtype=int)
    flat_neighbors = []
    neighbor_pointers = np.zeros(len(G) + 1, dtype=int)

    current_index = 0
    for i, j in extended_edge_vector:
        degrees[i] += 1
        flat_neighbors.append(j)

    for i in range(len(G)):
        neighbor_pointers[i] = current_index
        current_index += degrees[i]
    neighbor_pointers[-1] = current_index

    return extended_edge_vector, degrees, np.array(flat_neighbors, dtype=int), neighbor_pointers



def avg_degree(G: nx.Graph) -> float:
    return 2.0 * G.number_of_edges() / G.number_of_nodes()

def degree_dist_from_degrees(degrees: np.ndarray):

    hist = np.bincount(degrees.astype(int))
    pk = hist / hist.sum()
    k = np.arange(len(pk))
    return k, pk
def pad_and_stack_pk(pk_list):

    Kmax = max(len(pk) for pk in pk_list)
    H = np.zeros((len(pk_list), Kmax), dtype=float)
    for i, pk in enumerate(pk_list):
        H[i, :len(pk)] = pk
    return H

def degree_matching_params_standard(N: int, z_target: float, p_ws: float):

    # Watts–Strogatz: target mean degree ~= k (must be even, >=2)
    k_ws = int(round(z_target))
    k_ws = max(2, k_ws)
    if k_ws % 2 == 1:
        k_ws += 1
    k_ws = min(k_ws, N-1)

    # Barabási–Albert: mean degree ~= 2m
    m_ba = max(1, int(round(z_target / 2.0)))
    m_ba = min(m_ba, N-1)

    # Erdős–Rényi: mean degree ~= p*(N-1)
    p_er = float(z_target) / (N - 1.0)
    p_er = min(1.0, max(0.0, p_er))

    # Random-regular: degree d (must satisfy N*d even)
    d_rr = int(round(z_target))
    d_rr = min(max(d_rr, 0), N-1)
    if (N * d_rr) % 2 == 1:
        d_rr = d_rr + 1 if d_rr < N-1 else d_rr - 1

    return k_ws, p_ws, m_ba, p_er, d_rr


    
def lambda2_laplacian(G, dense_n_max=2500, tol=1e-8, maxiter=2000):
    """
    Robust numerical lambda2 for connected, undirected, unweighted graphs.
    - Dense eigendecomposition for small/moderate n (very reliable)
    - Sparse eigsh for larger sparse graphs (fast)
    """
    if not nx.is_connected(G):
        return 0.0

    # consistent node order
    nodelist = list(G.nodes())
    n = len(nodelist)

    L = nx.laplacian_matrix(G, nodelist=nodelist)  

    if n <= dense_n_max:
        Ld = L.toarray().astype(float)
        w = np.linalg.eigvalsh(Ld)
        w.sort()
        return float(w[1])

    try:
        import scipy.sparse.linalg as spla
        vals = spla.eigsh(L.asfptype(), k=2, which="SM",
                         tol=tol, maxiter=maxiter,
                         return_eigenvectors=False)
        vals = np.sort(np.real(vals))
        return float(vals[1])
    except Exception:
        return float(nx.algebraic_connectivity(G, tol=tol, method="tracemin"))
 
def structural_metrics(G):
    N = G.number_of_nodes()
    E = G.number_of_edges()
    k_mean = 2.0 * E / N
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G) 
    lam2 = lambda2_laplacian(G)
    return N, E, k_mean, C, L, lam2
