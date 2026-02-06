import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from numba import njit
import pandas as pd
import math
import os
import time



NEI_UNIFORM = 0
NEI_DEGPROB = 1
Q_FIXED = 0
Q_DYNAMIC = 1


@njit
def _choose_neighbor_uniform(neighbors):
    return neighbors[np.random.randint(neighbors.shape[0])]

@njit
def _choose_neighbor_degprob(neighbors, degrees):
    # sample neighbor proportional to degrees[neighbor]
    total = 0.0
    for idx in range(neighbors.shape[0]):
        total += degrees[neighbors[idx]]
    r = np.random.rand() * total
    c = 0.0
    for idx in range(neighbors.shape[0]):
        c += degrees[neighbors[idx]]
        if r <= c:
            return neighbors[idx]
    return neighbors[-1]






@njit
def single_run_general(flat_neighbors, neighbor_pointers, degrees,
                       max_steps, sigma, N, mu, p_i, p_e,
                       q_fixed, neighbor_mode=NEI_UNIFORM, q_mode=Q_FIXED):
    """
    neighbor_mode: NEI_UNIFORM or NEI_DEGPROB
    q_mode: Q_FIXED or Q_DYNAMIC
    """
    x = np.random.uniform(0.0, 1.0, N)
    variances = np.empty(max_steps + 1)
    variances[0] = np.var(x)

    for step in range(max_steps):
        new_x = x.copy()
        for i in range(N):
            start = neighbor_pointers[i]
            end   = neighbor_pointers[i + 1]
            if end <= start:
                continue
            neighbors = flat_neighbors[start:end]

            imitation = (np.random.rand() < p_i)
            exploration = (np.random.rand() < p_e)

            if imitation:
                if neighbor_mode == 0:
                    j = _choose_neighbor_uniform(neighbors)
                else:
                    j = _choose_neighbor_degprob(neighbors, degrees)

                if q_mode == 0:
                    q = q_fixed
                else:
                    di = degrees[i]
                    dj = degrees[j]
                    denom = di + dj
                    q = (dj / denom) 

                new_x[i] = (1.0 - q) * x[i] + q * x[j]

            if exploration:
                new_x[i] += np.random.normal(mu, sigma)

        x = new_x
        variances[step + 1] = np.var(x)

    return variances

