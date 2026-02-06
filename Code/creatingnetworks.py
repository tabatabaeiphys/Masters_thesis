import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from numba import njit
import pandas as pd
import math
import os
import time







def star_backbone_params_degree_matched(
    N: int,
    z_target: float,
    l_fixed: int,
    require_even_b: bool = True,
    enforce_connected_backbone: bool = True,
    tol: float = 1e-12,
):

    l = int(l_fixed)
    if l < 1:
        raise ValueError("l_fixed must be >= 1.")

    if N % (l + 1) != 0:
        raise ValueError(
            f"N={N} not divisible by (l+1)={l+1}; cannot keep all stars with l={l}."
        )

    s = N // (l + 1)
    if s < 2:
        raise ValueError("Need at least s>=2 hubs.")

    b_ideal = z_target * (l + 1) - 2.0 * l

    b_min = 0
    b_max = s - 1

    if enforce_connected_backbone and s >= 3:
        b_min = 2  

    if b_ideal < b_min - tol:
        raise ValueError(
            f"Target z={z_target} implies b≈{b_ideal:.3f}, but need b>={b_min} for s={s}."
        )
    if b_ideal > b_max + tol:
        raise ValueError(
            f"Target z={z_target} implies b≈{b_ideal:.3f}, but max possible is s-1={b_max}."
        )

    b = int(round(b_ideal))

    if require_even_b and (b % 2 == 1):
        down = b - 1
        up = b + 1
        candidates = []
        for bb in (down, up):
            if bb % 2 == 0 and (b_min <= bb <= b_max):
                candidates.append(bb)
        if not candidates:
            raise ValueError(
                f"Could not find an even b within [{b_min},{b_max}] near b_ideal={b_ideal:.3f}."
            )
        b = min(candidates, key=lambda bb: abs(bb - b_ideal))

    if b < b_min or b > b_max:
        raise ValueError(
            f"Rounded b={b} out of feasible range [{b_min},{b_max}] for s={s}."
        )
    if require_even_b and (b % 2 == 1):
        raise ValueError("Internal error: b ended odd despite require_even_b=True.")

    achieved = (2.0 * l + b) / (l + 1)
    return s, l, b, achieved, b_ideal


def build_stars_with_backbone(
    N: int,
    s: int,
    l: int,
    b: int,
    seed: int | None = None,
):

    if s * (l + 1) != N:
        raise ValueError("N must equal s*(l+1).")
    if not (0 <= b <= s - 1):
        raise ValueError("Backbone degree b must satisfy 0 <= b <= s-1.")
    if s >= 3 and b < 2:
        raise ValueError("For s>=3, use b>=2 to keep the backbone connected.")
    if s >= 3 and (b % 2 == 1):
        raise ValueError("For WS ring-lattice backbone, b must be even when s>=3.")

    G = nx.Graph()

    hubs = list(range(s))
    G.add_nodes_from(hubs, kind="hub")

    if s == 1:
        backbone = nx.empty_graph(1)
    elif s == 2:
        backbone = nx.Graph()
        backbone.add_nodes_from([0, 1])
        if b == 1:
            backbone.add_edge(0, 1)
    else:
        backbone = nx.watts_strogatz_graph(s, b, 0.0, seed=seed)

    for u, v in backbone.edges():
        G.add_edge(u, v, kind="backbone")

    node_id = s
    for h in hubs:
        for _ in range(l):
            G.add_node(node_id, kind="leaf")
            G.add_edge(h, node_id, kind="leaf")
            node_id += 1

    if G.number_of_nodes() != N:
        raise RuntimeError("Node count mismatch after construction.")
    if not nx.is_connected(G):
        raise RuntimeError("Constructed star-backbone graph is not connected (unexpected).")

    return G


def create_star_backbone_degree_matched(
    N: int,
    z_target: float,
    l_fixed: int,
    seed: int | None = None,
    require_even_b: bool = True,
):
    """
    Convenience wrapper:
      - compute degree-matched parameters for fixed l
      - build the graph
      - return (G, info)
    """
    s, l, b, achieved, b_ideal = star_backbone_params_degree_matched(
        N=N,
        z_target=z_target,
        l_fixed=l_fixed,
        require_even_b=require_even_b,
        enforce_connected_backbone=True
    )

    G = build_stars_with_backbone(N=N, s=s, l=l, b=b, seed=seed)

    info = {
        "num_stars": s,
        "leaves_per_star": l,
        "backbone_degree": b,
        "b_ideal": b_ideal,
        "k_target": z_target,
        "k_achieved_formula": achieved,
        "k_mean_graph": 2.0 * G.number_of_edges() / G.number_of_nodes(),
    }
    return G, info







def clique_addedge_params(N: int, clique_size: int, z_target: float, backbone: str = "ring"):

    s = int(clique_size)
    if s < 2:
        raise ValueError("clique_size must be >= 2.")
    if N % s != 0:
        raise ValueError(f"N={N} must be divisible by clique_size={s} to form equal cliques.")

    c = N // s  
    E_cliques = c * (s * (s - 1) // 2)

    if backbone == "ring":
        E_backbone = 0 if c == 1 else (1 if c == 2 else c)
    elif backbone == "line":
        E_backbone = 0 if c == 1 else (c - 1)
    else:
        raise ValueError("backbone must be 'ring' or 'line'.")

    E_max = N * (N - 1) // 2

    # Minimum mean degree with fixed clique size 
    z_min = 2.0 * (E_cliques + E_backbone) / N
    z_max = 2.0 * E_max / N  # = N-1

    E_target_real = z_target * N / 2.0
    E_target = int(round(E_target_real))

    E_add_needed = E_target - (E_cliques + E_backbone)

    feasible = True
    reasons = []
    if E_add_needed < 0:
        feasible = False
        reasons.append(
            f"Target z={z_target} implies fewer edges than the base cliques+backbone already have "
            f"(z_min≈{z_min:.3f}). You cannot reduce degree by adding edges."
        )
    if E_target > E_max:
        feasible = False
        reasons.append("Target exceeds complete graph edge count.")

    return {
        "N": N,
        "clique_size": s,
        "n_cliques": c,
        "backbone": backbone,
        "E_cliques": E_cliques,
        "E_backbone": E_backbone,
        "E_base": E_cliques + E_backbone,
        "z_min": z_min,
        "z_max": z_max,
        "E_target_real": E_target_real,
        "E_target_int": E_target,
        "E_add_needed": E_add_needed,
        "feasible": feasible,
        "reasons": reasons,
    }






def build_cliques_with_added_edges(
    N: int,
    clique_size: int,
    z_target: float,
    backbone: str = "ring",
    seed: int | None = None,
    tidy_mode: str = "pairwise_roundrobin",
):

    rng = np.random.default_rng(seed)

    params = clique_addedge_params(N, clique_size, z_target, backbone=backbone)
    if not params["feasible"]:
        raise ValueError("Infeasible target.\n" + "\n".join(params["reasons"]))

    s = params["clique_size"]
    c = params["n_cliques"]
    E_add_needed = params["E_add_needed"]

    G = nx.Graph()
    clique_nodes = []
    for r in range(c):
        nodes = list(range(r * s, (r + 1) * s))
        clique_nodes.append(nodes)
        for i in range(s):
            for j in range(i + 1, s):
                G.add_edge(nodes[i], nodes[j], kind="intra")

    connectors = [nodes[0] for nodes in clique_nodes]

    if backbone == "ring":
        if c == 2:
            G.add_edge(connectors[0], connectors[1], kind="backbone")
        elif c >= 3:
            for r in range(c):
                G.add_edge(connectors[r], connectors[(r + 1) % c], kind="backbone")
    elif backbone == "line":
        for r in range(c - 1):
            G.add_edge(connectors[r], connectors[r + 1], kind="backbone")

    def clique_index(u):  
        return u // s

    def try_add(u, v):
        if u == v:
            return False
        if clique_index(u) == clique_index(v):
            return False 
        if G.has_edge(u, v):
            return False
        G.add_edge(u, v, kind="added")
        return True

    added = 0

    if tidy_mode == "pairwise_roundrobin":
        node_ptr = [0] * c  
        pairs = [(a, b) for a in range(c) for b in range(a + 1, c)]
        pidx = 0
        max_tries = 10 * max(1, E_add_needed)

        tries = 0
        while added < E_add_needed and tries < max_tries:
            a, b = pairs[pidx % len(pairs)]
            pidx += 1

            ua = clique_nodes[a][node_ptr[a] % s]
            ub = clique_nodes[b][node_ptr[b] % s]
            node_ptr[a] += 1
            node_ptr[b] += 1

            if try_add(ua, ub):
                added += 1
            tries += 1

        if added < E_add_needed:
            tidy_mode = "random"

    if tidy_mode == "random":
        max_tries = 50 * max(1, E_add_needed)
        tries = 0
        while added < E_add_needed and tries < max_tries:
            u = int(rng.integers(0, N))
            v = int(rng.integers(0, N))
            if try_add(u, v):
                added += 1
            tries += 1
        if added < E_add_needed:
            raise RuntimeError(
                f"Could not add enough inter-clique edges (added {added}/{E_add_needed}). "
                "This can happen only if the graph is near-complete."
            )

    if not nx.is_connected(G):
        raise RuntimeError("Unexpected: graph is not connected after adding backbone.")

    info = dict(params)
    info.update({
        "E_final": G.number_of_edges(),
        "k_mean_graph": 2.0 * G.number_of_edges() / G.number_of_nodes(),
        "E_added_actual": added,
        "tidy_mode_used": tidy_mode,
    })
    return G, info








def clique_gateway_params_degree_matched(
    N: int,
    clique_size: int,
    z_target: float,
    backbone: str = "ring",
    require_exact: bool = False,
):
    
    s = int(clique_size)
    if s < 2:
        raise ValueError("clique_size must be >= 2.")
    if N % s != 0:
        raise ValueError(f"N={N} must be divisible by clique_size={s}.")

    c = N // s  
    E_cliques = c * (s * (s - 1) // 2)

    if backbone == "ring":
        E_backbone = 0 if c == 1 else (1 if c == 2 else c)
    elif backbone == "line":
        E_backbone = 0 if c == 1 else (c - 1)
    else:
        raise ValueError("backbone must be 'ring' or 'line'.")

    E_target_real = z_target * N / 2.0
    E_target = int(round(E_target_real)) if not require_exact else int(E_target_real)

    E_base = E_cliques + E_backbone

    if E_target < E_base:
        z_min = 2.0 * E_base / N
        raise ValueError(
            f"Target z={z_target} too small for clique_size={s}. "
            f"Base already has z_min≈{z_min:.3f}."
        )

    E_gate_total_possible = c * (c - 1) // 2
    E_extra_max = E_gate_total_possible - E_backbone
    if E_extra_max < 0:
        E_extra_max = 0

    E_max_reachable = E_base + E_extra_max
    if E_target > E_max_reachable:
        z_max_reachable = 2.0 * E_max_reachable / N
        raise ValueError(
            f"Target z={z_target} is infeasible under gateway-only constraint. "
            f"With N={N}, clique_size={s} (c={c}), backbone={backbone}, "
            f"max reachable mean degree is z_max≈{z_max_reachable:.3f}. "
            f"To reach higher z, allow multiple connectors per clique or allow added edges between non-gateway nodes."
        )

    E_add_needed = E_target - E_base

    return {
        "N": N,
        "clique_size": s,
        "n_cliques": c,
        "backbone": backbone,
        "E_cliques": E_cliques,
        "E_backbone": E_backbone,
        "E_base": E_base,
        "E_target_real": E_target_real,
        "E_target_int": E_target,
        "E_add_needed": E_add_needed,
        "E_extra_max": E_extra_max,
    }




def build_cliques_gateway_only(
    N: int,
    clique_size: int,
    z_target: float,
    backbone: str = "ring",
    seed: int | None = None,
    require_exact: bool = False,
):
    """
    Build gateway-only (star-like) clique network:
      - Disjoint cliques of fixed size
      - One gateway per clique
      - Backbone on gateways (ring/line)
      - Additional edges ONLY between gateways, added in a tidy round-robin way

    Returns: (G, info)
    """
    rng = np.random.default_rng(seed)
    params = clique_gateway_params_degree_matched(
        N=N, clique_size=clique_size, z_target=z_target,
        backbone=backbone, require_exact=require_exact
    )

    s = params["clique_size"]
    c = params["n_cliques"]
    E_add_needed = params["E_add_needed"]

    G = nx.Graph()
    cliques = []
    for r in range(c):
        nodes = list(range(r * s, (r + 1) * s))
        cliques.append(nodes)
        for i in range(s):
            for j in range(i + 1, s):
                G.add_edge(nodes[i], nodes[j], kind="intra")

    gateways = [nodes[0] for nodes in cliques]

    if backbone == "ring":
        if c == 2:
            G.add_edge(gateways[0], gateways[1], kind="backbone")
        elif c >= 3:
            for r in range(c):
                G.add_edge(gateways[r], gateways[(r + 1) % c], kind="backbone")
    elif backbone == "line":
        for r in range(c - 1):
            G.add_edge(gateways[r], gateways[r + 1], kind="backbone")

    if E_add_needed > 0:
        pairs = [(a, b) for a in range(c) for b in range(a + 1, c)]
        rng.shuffle(pairs)  

        added = 0
        idx = 0
        while added < E_add_needed:
            if idx >= len(pairs):
                raise RuntimeError("Ran out of gateway pairs; should not happen due to feasibility check.")
            a, b = pairs[idx]
            idx += 1
            u, v = gateways[a], gateways[b]
            if not G.has_edge(u, v):
                G.add_edge(u, v, kind="added")
                added += 1
    else:
        added = 0

    if G.number_of_nodes() != N:
        raise RuntimeError("Node count mismatch after construction.")
    if not nx.is_connected(G):
        raise RuntimeError("Graph is not connected (unexpected; backbone should guarantee it).")

    info = dict(params)
    info.update({
        "E_final": G.number_of_edges(),
        "E_added_actual": added,
        "k_mean_graph": 2.0 * G.number_of_edges() / G.number_of_nodes(),
        "gateway_only": True,
    })
    return G, info

_ENSURE_COUNTER = {}
def ensure_connected(make_graph, name="G", max_tries=200, seed0=None):

    if seed0 is None:
        c = _ENSURE_COUNTER.get(name, 0)
        seed0 = (c + 1) * 10_000
        _ENSURE_COUNTER[name] = c + 1

    for t in range(max_tries):
        seed = seed0 + t
        try:
            G = make_graph(seed)     
        except TypeError:
            G = make_graph()         

        if nx.is_connected(G):
            return G

    raise RuntimeError(f"{name}: could not generate a connected graph in {max_tries} tries.")
