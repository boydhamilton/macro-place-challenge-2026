"""
Genetic Algorithm Macro Placer with CARDS-style Refinement

Combines ideas from two papers:

Paper 1 — CARDS Framework (post-legalization refinement):
  - Vacant area centroid movement: divide canvas into 16×16 grid, move macros
    toward nearest vacant centroid to reduce crowding (density/congestion)
  - 16-type boundary flipping: enumerate all orientation combos for adjacent
    macro pairs, pick the one minimizing wirelength on the highest-cost wire

Paper 2 — Zhang et al. Genetic Algorithm:
  - Population-based global search avoids local optima of pure SA
  - Large macro edge placement: macros > 2× average area biased to canvas boundary
  - Fitness = HPWL + penalties (adapted to proxy cost framework)
  - GA operators: crossover swaps macro positions; mutation perturbs via SA moves

Combined strategy:
  GA for global ordering/edge placement → CARDS vacant centroid post-processing
  Proxy cost formula (WL + 0.5×density + 0.5×congestion) drives fitness.
  We use fast HPWL as the inner-loop fitness, compute full proxy only at the end.

Usage:
    uv run evaluate submissions/genetic_placer/placer.py
    uv run evaluate submissions/genetic_placer/placer.py --all
    uv run evaluate submissions/genetic_placer/placer.py -b ibm01
"""

import math
import random
import numpy as np
import torch
from pathlib import Path

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Data loading helpers (mirrors will_seed pattern)
# ---------------------------------------------------------------------------

def _load_plc(name):
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45.get(name)
    if d:
        base = (Path("external/MacroPlacement/Flows/NanGate45")
                / d / "netlist" / "output_CT_Grouping")
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(
                str(base / "netlist.pb.txt"), str(base / "initial.plc")
            )
            return plc
    return None


def _extract_edges(benchmark, plc):
    """Extract weighted macro-to-macro edges from netlist."""
    name_to_bidx = {}
    for bidx, idx in enumerate(benchmark.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx

    edge_dict = {}
    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) >= 2:
            ml = sorted(macros)
            w = 1.0 / (len(ml) - 1)
            for i in range(len(ml)):
                for j in range(i + 1, len(ml)):
                    key = (ml[i], ml[j])
                    edge_dict[key] = edge_dict.get(key, 0) + w

    if not edge_dict:
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float32)

    edges = np.array(list(edge_dict.keys()), dtype=np.int32)
    weights = np.array([edge_dict[tuple(e)] for e in edges], dtype=np.float32)
    return edges, weights


def _build_neighbors(n, edges):
    """Build adjacency list from edge list."""
    neighbors = [[] for _ in range(n)]
    for i, j in edges:
        neighbors[i].append(int(j))
        neighbors[j].append(int(i))
    return neighbors


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def _make_sep(sizes):
    """Precompute separation matrices: sep_x[i,j] = (w_i + w_j) / 2."""
    w = sizes[:, 0]
    h = sizes[:, 1]
    sep_x = (w[:, np.newaxis] + w[np.newaxis, :]) / 2.0  # [n, n]
    sep_y = (h[:, np.newaxis] + h[np.newaxis, :]) / 2.0
    return sep_x, sep_y


def _check_overlap(idx, pos, sep_x, sep_y, n, gap=0.05):
    """O(n) overlap check: does macro idx overlap any of the first n macros?"""
    dx = np.abs(pos[idx, 0] - pos[:n, 0])
    dy = np.abs(pos[idx, 1] - pos[:n, 1])
    ov = (dx < sep_x[idx, :n] + gap) & (dy < sep_y[idx, :n] + gap)
    ov[idx] = False
    return bool(ov.any())


def _hpwl(pos, edges, weights):
    """Weighted pairwise HPWL over macro-to-macro edges."""
    if len(edges) == 0:
        return 0.0
    dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
    dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
    return float((weights * (dx + dy)).sum())


def _fast_density_cost(pos, sizes, n, cw, ch, grid_size=10):
    """
    Fast density cost proxy (mirrors plc_client density formula).

    Rasterizes macros onto a grid_size×grid_size grid using fractional overlap,
    then returns 0.5 × average of the top 10% most-occupied cells.
    """
    cell_w = cw / grid_size
    cell_h = ch / grid_size
    cell_area = cell_w * cell_h
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)

    for i in range(n):
        hw, hh = sizes[i, 0] / 2, sizes[i, 1] / 2
        x0, x1 = pos[i, 0] - hw, pos[i, 0] + hw
        y0, y1 = pos[i, 1] - hh, pos[i, 1] + hh

        col_lo = max(0, int(x0 / cell_w))
        col_hi = min(grid_size - 1, int(x1 / cell_w))
        row_lo = max(0, int(y0 / cell_h))
        row_hi = min(grid_size - 1, int(y1 / cell_h))

        cols = np.arange(col_lo, col_hi + 1)
        rows = np.arange(row_lo, row_hi + 1)
        ov_x = np.maximum(0.0, np.minimum(x1, (cols + 1) * cell_w) - np.maximum(x0, cols * cell_w))
        ov_y = np.maximum(0.0, np.minimum(y1, (rows + 1) * cell_h) - np.maximum(y0, rows * cell_h))
        grid[row_lo:row_hi + 1, col_lo:col_hi + 1] += np.outer(ov_y, ov_x) / cell_area

    flat = grid.ravel()
    n_top = max(1, grid_size * grid_size // 10)
    top10 = np.partition(flat, -n_top)[-n_top:]
    return 0.5 * float(top10.mean())


def _fast_congestion_cost(pos, edges, weights, cw, ch, grid_size=10):
    """
    Cut-based congestion proxy.

    For each of the (grid_size-1) vertical and horizontal grid boundaries,
    sums the weight of nets whose bounding box spans that cut. Normalizes by
    total net weight. Returns abu(top 10%) of the 2*(grid_size-1) cut values.
    """
    if len(edges) == 0:
        return 0.0
    total_w = float(weights.sum())
    if total_w < 1e-9:
        return 0.0

    cell_w = cw / grid_size
    cell_h = ch / grid_size

    x0 = np.minimum(pos[edges[:, 0], 0], pos[edges[:, 1], 0])
    x1 = np.maximum(pos[edges[:, 0], 0], pos[edges[:, 1], 0])
    y0 = np.minimum(pos[edges[:, 0], 1], pos[edges[:, 1], 1])
    y1 = np.maximum(pos[edges[:, 0], 1], pos[edges[:, 1], 1])

    v_cuts = np.array([
        float(weights[(x0 < k * cell_w) & (x1 > k * cell_w)].sum())
        for k in range(1, grid_size)
    ]) / total_w

    h_cuts = np.array([
        float(weights[(y0 < k * cell_h) & (y1 > k * cell_h)].sum())
        for k in range(1, grid_size)
    ]) / total_w

    all_cuts = np.concatenate([v_cuts, h_cuts])
    n_top = max(1, len(all_cuts) // 10)
    top10 = np.partition(all_cuts, -n_top)[-n_top:]
    return float(top10.mean())


def _proxy_fitness(pos, edges, weights, sizes, n, cw, ch):
    """
    Full proxy fitness matching the evaluation metric:
      normalized_HPWL + 0.5 * density + 0.5 * congestion

    HPWL is normalized by (cw + ch) * total_weight to be commensurate
    with the density and congestion terms (both in [0, ~1]).
    """
    total_w = float(weights.sum()) if len(weights) > 0 else 1.0
    hpwl_norm = _hpwl(pos, edges, weights) / max((cw + ch) * total_w, 1e-9)
    density = _fast_density_cost(pos, sizes, n, cw, ch)
    congestion = _fast_congestion_cost(pos, edges, weights, cw, ch)
    return hpwl_norm + 0.5 * density + 0.5 * congestion


# ---------------------------------------------------------------------------
# Legalization: spiral search with minimum displacement
# ---------------------------------------------------------------------------

def _legalize(pos, movable, sizes, half_w, half_h, cw, ch, sep_x, sep_y):
    """
    Place each macro in a legal position using spiral search.
    Processes largest macros first for better packing.
    Minimizes displacement from the input positions.
    """
    n = len(pos)
    order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        # Check if already conflict-free
        if placed.any():
            p = placed.copy()
            p[idx] = False
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            if not ((dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & p).any():
                placed[idx] = True
                continue

        # Spiral search for nearest legal position
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
        best_p = np.clip(legal[idx], [half_w[idx], half_h[idx]],
                         [cw - half_w[idx], ch - half_h[idx]])
        best_d = float("inf")

        for r in range(1, 200):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = float(np.clip(pos[idx, 0] + dxm * step, half_w[idx], cw - half_w[idx]))
                    cy = float(np.clip(pos[idx, 1] + dym * step, half_h[idx], ch - half_h[idx]))
                    if placed.any():
                        p = placed.copy()
                        p[idx] = False
                        dx = np.abs(cx - legal[:, 0])
                        dy = np.abs(cy - legal[:, 1])
                        if ((dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & p).any():
                            continue
                    d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    if d < best_d:
                        best_d = d
                        best_p = np.array([cx, cy])
                        found = True
            if found:
                break

        legal[idx] = best_p
        placed[idx] = True

    return legal


# ---------------------------------------------------------------------------
# Initialization strategies
# ---------------------------------------------------------------------------

def _greedy_row_place(movable_idx, sizes, half_w, half_h, cw, ch, init_pos, sort_key=None):
    """Shelf-packing placement with configurable sort order."""
    gap = 0.001
    indices = list(movable_idx)
    indices.sort(key=(sort_key if sort_key is not None else lambda i: -sizes[i, 1]))

    pos = init_pos.copy()
    cursor_x, cursor_y, row_h = 0.0, 0.0, 0.0

    for idx in indices:
        w, h = float(sizes[idx, 0]), float(sizes[idx, 1])
        if cursor_x + w > cw:
            cursor_x = 0.0
            cursor_y += row_h + gap
            row_h = 0.0
        if cursor_y + h > ch:
            pos[idx] = [half_w[idx], half_h[idx]]
            continue
        pos[idx] = [cursor_x + w / 2, cursor_y + h / 2]
        cursor_x += w + gap
        row_h = max(row_h, h)

    return pos


def _edge_biased_place(movable_idx, sizes, half_w, half_h, cw, ch, init_pos, areas):
    """
    Large macro edge placement (Zhang et al.).

    Macros with area > 2× average are placed along canvas boundary (bottom/top
    rows), freeing the interior for smaller macros. Reduces density and
    congestion in the central routing area.
    """
    gap = 0.001
    movable_list = list(movable_idx)
    avg_area = float(areas[movable_list].mean())
    large_thresh = 2.0 * avg_area

    large = sorted([i for i in movable_list if areas[i] >= large_thresh],
                   key=lambda i: -areas[i])
    small = sorted([i for i in movable_list if areas[i] < large_thresh],
                   key=lambda i: -sizes[i, 1])

    pos = init_pos.copy()
    placed_set = set()
    bottom_h = 0.0
    top_h = 0.0

    # Bottom row: first half of large macros
    cursor_x = 0.0
    for idx in large[: len(large) // 2 + 1]:
        w, h = float(sizes[idx, 0]), float(sizes[idx, 1])
        if cursor_x + w > cw:
            break
        pos[idx] = [cursor_x + w / 2, half_h[idx]]
        cursor_x += w + gap
        bottom_h = max(bottom_h, h)
        placed_set.add(idx)

    # Top row: remaining large macros
    cursor_x = 0.0
    for idx in large:
        if idx in placed_set:
            continue
        w, h = float(sizes[idx, 0]), float(sizes[idx, 1])
        if cursor_x + w > cw:
            small.append(idx)  # Overflow → treat as small
            continue
        pos[idx] = [cursor_x + w / 2, ch - half_h[idx]]
        cursor_x += w + gap
        top_h = max(top_h, h)
        placed_set.add(idx)

    # Middle rows: small macros
    start_y = bottom_h + gap
    end_y = (ch - top_h - gap) if top_h > 0 else ch
    cursor_x, cursor_y, row_h = 0.0, start_y, 0.0

    for idx in small:
        w, h = float(sizes[idx, 0]), float(sizes[idx, 1])
        if cursor_x + w > cw:
            cursor_x = 0.0
            cursor_y += row_h + gap
            row_h = 0.0
        if cursor_y + h > end_y:
            pos[idx] = [cw / 2, ch / 2]  # Fallback: canvas center
            continue
        pos[idx] = [cursor_x + w / 2, cursor_y + h / 2]
        cursor_x += w + gap
        row_h = max(row_h, h)

    return pos


# ---------------------------------------------------------------------------
# SA-style mutation (used inside GA and for final polish)
# ---------------------------------------------------------------------------

def _sa_evolve(pos, movable_idx, neighbors, sizes, half_w, half_h,
               cw, ch, sep_x, sep_y, n, edges, weights,
               n_steps, T_start, T_end):
    """
    SA with overlap-rejection (no full re-legalization needed).

    Move types:
      - Gaussian shift (50%): random perturbation toward lower temperature
      - Swap (25%): swap positions of two macros
      - Neighbor pull (25%): move toward a connected macro
    """
    if len(movable_idx) == 0:
        return pos

    pos = pos.copy()
    current_cost = _hpwl(pos, edges, weights)
    best_pos = pos.copy()
    best_cost = current_cost

    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        T = T_start * (T_end / max(T_start, 1e-12)) ** frac

        move = random.random()
        i = random.choice(movable_idx)
        old_x, old_y = pos[i, 0], pos[i, 1]

        if move < 0.5:
            # Gaussian shift
            scale = T * (0.3 + 0.7 * (1.0 - frac))
            pos[i, 0] = float(np.clip(pos[i, 0] + random.gauss(0, scale),
                                       half_w[i], cw - half_w[i]))
            pos[i, 1] = float(np.clip(pos[i, 1] + random.gauss(0, scale),
                                       half_h[i], ch - half_h[i]))

        elif move < 0.75:
            # Swap two macros (prefer connected neighbor for quality)
            cands = [j for j in neighbors[i] if j in movable_idx] if neighbors[i] else []
            j = (random.choice(cands) if cands and random.random() < 0.6
                 else random.choice(movable_idx))
            if i == j:
                continue

            old_jx, old_jy = pos[j, 0], pos[j, 1]
            pos[i, 0] = float(np.clip(old_jx, half_w[i], cw - half_w[i]))
            pos[i, 1] = float(np.clip(old_jy, half_h[i], ch - half_h[i]))
            pos[j, 0] = float(np.clip(old_x, half_w[j], cw - half_w[j]))
            pos[j, 1] = float(np.clip(old_y, half_h[j], ch - half_h[j]))

            if _check_overlap(i, pos, sep_x, sep_y, n) or \
               _check_overlap(j, pos, sep_x, sep_y, n):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                pos[j, 0] = old_jx; pos[j, 1] = old_jy
                continue

            new_cost = _hpwl(pos, edges, weights)
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos = pos.copy()
            else:
                pos[i, 0] = old_x; pos[i, 1] = old_y
                pos[j, 0] = old_jx; pos[j, 1] = old_jy
            continue

        else:
            # Pull toward connected neighbor
            if not neighbors[i]:
                continue
            j = random.choice(neighbors[i])
            alpha = random.uniform(0.05, 0.3)
            pos[i, 0] = float(np.clip(pos[i, 0] + alpha * (pos[j, 0] - pos[i, 0]),
                                       half_w[i], cw - half_w[i]))
            pos[i, 1] = float(np.clip(pos[i, 1] + alpha * (pos[j, 1] - pos[i, 1]),
                                       half_h[i], ch - half_h[i]))

        # Overlap check for shift / neighbor-pull moves
        if _check_overlap(i, pos, sep_x, sep_y, n):
            pos[i, 0] = old_x; pos[i, 1] = old_y
            continue

        new_cost = _hpwl(pos, edges, weights)
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_pos = pos.copy()
        else:
            pos[i, 0] = old_x; pos[i, 1] = old_y

    return best_pos


# ---------------------------------------------------------------------------
# GA crossover
# ---------------------------------------------------------------------------

def _crossover(pos_a, pos_b, movable_idx, half_w, half_h, cw, ch, sep_x, sep_y, n):
    """
    Position crossover: for each macro, randomly inherit position from parent B.
    Reverts any swap that creates an overlap (maintains legality without re-legalizing).
    Processes macros in shuffled order to reduce order bias.
    """
    child = pos_a.copy()
    indices = list(movable_idx)
    random.shuffle(indices)

    for idx in indices:
        if random.random() < 0.5:
            old = child[idx].copy()
            child[idx, 0] = float(np.clip(pos_b[idx, 0], half_w[idx], cw - half_w[idx]))
            child[idx, 1] = float(np.clip(pos_b[idx, 1], half_h[idx], ch - half_h[idx]))
            if _check_overlap(idx, child, sep_x, sep_y, n):
                child[idx] = old  # Revert: cannot inherit from B without overlap

    return child


# ---------------------------------------------------------------------------
# CARDS-style vacant centroid movement (post-legalization refinement)
# ---------------------------------------------------------------------------

"""
def _vacant_centroid_move(pos, movable_mask, sizes, half_w, half_h,
                          cw, ch, sep_x, sep_y, n, grid_size=16, step_frac=0.25):
    
    CARDS vacant area centroid finding and movement.

    Divides the canvas into a grid_size × grid_size grid, identifies cells
    not occupied by any macro, and moves each macro toward the centroid of
    the nearest vacant cluster. This reduces density in crowded areas.

    Movement distance scales with proximity (close macros move less),
    preserving relative ordering while opening routing space.
    
    cell_w = cw / grid_size
    cell_h = ch / grid_size

    # Mark occupied grid cells (rasterize each hard macro onto the grid)
    occupied = np.zeros((grid_size, grid_size), dtype=bool)
    for i in range(n):
        hw, hh = sizes[i, 0] / 2, sizes[i, 1] / 2
        col_lo = max(0, int((pos[i, 0] - hw) / cell_w))
        col_hi = min(grid_size - 1, int((pos[i, 0] + hw) / cell_w))
        row_lo = max(0, int((pos[i, 1] - hh) / cell_h))
        row_hi = min(grid_size - 1, int((pos[i, 1] + hh) / cell_h))
        occupied[row_lo: row_hi + 1, col_lo: col_hi + 1] = True

    # Collect vacant cell centers
    vacant = []
    for r in range(grid_size):
        for c in range(grid_size):
            if not occupied[r, c]:
                vacant.append([(c + 0.5) * cell_w, (r + 0.5) * cell_h])

    if not vacant:
        return pos

    vacant_arr = np.array(vacant)  # [K, 2]
    new_pos = pos.copy()
    movable_idx = np.where(movable_mask[:n])[0]

    for idx in movable_idx:
        px, py = pos[idx, 0], pos[idx, 1]

        # Find nearest vacant centroid
        dists = np.sqrt((vacant_arr[:, 0] - px) ** 2 + (vacant_arr[:, 1] - py) ** 2)
        nearest = vacant_arr[int(np.argmin(dists))]

        dx = nearest[0] - px
        dy = nearest[1] - py
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < 1e-6:
            continue

        # Scale step by distance: farther macros move more (CARDS convention)
        alpha = step_frac * min(1.0, dist / (max(cw, ch) * 0.1))

        old_x, old_y = new_pos[idx, 0], new_pos[idx, 1]
        new_pos[idx, 0] = float(np.clip(px + alpha * dx, half_w[idx], cw - half_w[idx]))
        new_pos[idx, 1] = float(np.clip(py + alpha * dy, half_h[idx], ch - half_h[idx]))

        # Revert if the move creates an overlap
        if _check_overlap(idx, new_pos, sep_x, sep_y, n):
            new_pos[idx, 0] = old_x
            new_pos[idx, 1] = old_y

    return new_pos
"""

# new, correct implementation that finds centroids of connected vacant regions (not individual cells) and uses the CARDS alpha formula based on Dmax/Dmin to scale movement by proximity to the centroid cluster
def _vacant_centroid_move(pos, movable_mask, sizes, half_w, half_h,
                          cw, ch, sep_x, sep_y, n, grid_size=16, step_frac=0.25):
    cell_w = cw / grid_size
    cell_h = ch / grid_size

    occupied = np.zeros((grid_size, grid_size), dtype=bool)
    for i in range(n):
        hw, hh = sizes[i, 0] / 2, sizes[i, 1] / 2
        col_lo = max(0, int((pos[i, 0] - hw) / cell_w))
        col_hi = min(grid_size - 1, int((pos[i, 0] + hw) / cell_w))
        row_lo = max(0, int((pos[i, 1] - hh) / cell_h))
        row_hi = min(grid_size - 1, int((pos[i, 1] + hh) / cell_h))
        occupied[row_lo:row_hi + 1, col_lo:col_hi + 1] = True

    vacant_centroids = _find_vacant_region_centroids(occupied, cell_w, cell_h, grid_size)
    if not vacant_centroids:
        return pos

    vacant_arr = np.array(vacant_centroids)
    new_pos = pos.copy()
    movable_idx = np.where(movable_mask[:n])[0]

    # For each movable macro, find its nearest centroid and compute D
    macro_positions = pos[movable_idx]  # [M, 2]
    # [K, M] distance matrix
    dists_matrix = np.sqrt(
        (vacant_arr[:, 0:1] - macro_positions[:, 0]) ** 2 +
        (vacant_arr[:, 1:2] - macro_positions[:, 1]) ** 2
    )
    nearest_centroid_per_macro = np.argmin(dists_matrix, axis=0)  # [M]
    dist_per_macro = dists_matrix[nearest_centroid_per_macro, np.arange(len(movable_idx))]

    # FIX 3: compute Dmax/Dmin per centroid group
    alpha_per_macro = np.zeros(len(movable_idx))
    for c_idx in range(len(vacant_centroids)):
        members = np.where(nearest_centroid_per_macro == c_idx)[0]
        if len(members) == 0:
            continue
        D_vals = dist_per_macro[members]
        Dmax, Dmin = D_vals.max(), D_vals.min()
        denom = max(Dmax - Dmin, 1e-6)
        # Paper eq 1: closer macros (small D) get larger step
        alpha_per_macro[members] = step_frac * (Dmax - D_vals) / denom

    # FIX 3: process farthest-first to open corridors before moving inner macros
    order = np.argsort(-dist_per_macro)

    # FIX 2: relaxed gap for refinement (macros are already legal, just nudging)
    refinement_gap = 0.01

    for local_i in order:
        idx = movable_idx[local_i]
        px, py = new_pos[idx, 0], new_pos[idx, 1]  # use updated pos, not original
        c_idx = nearest_centroid_per_macro[local_i]
        nearest = vacant_arr[c_idx]

        dx = nearest[0] - px
        dy = nearest[1] - py
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 1e-6:
            continue

        alpha = alpha_per_macro[local_i]
        old_x, old_y = new_pos[idx, 0], new_pos[idx, 1]
        new_pos[idx, 0] = float(np.clip(px + alpha * dx / dist, half_w[idx], cw - half_w[idx]))
        new_pos[idx, 1] = float(np.clip(py + alpha * dy / dist, half_h[idx], ch - half_h[idx]))

        # FIX 2: relaxed overlap threshold for refinement
        dx_ov = np.abs(new_pos[idx, 0] - new_pos[:n, 0])
        dy_ov = np.abs(new_pos[idx, 1] - new_pos[:n, 1])
        ov = (dx_ov < sep_x[idx, :n] + refinement_gap) & (dy_ov < sep_y[idx, :n] + refinement_gap)
        ov[idx] = False
        if ov.any():
            new_pos[idx, 0] = old_x
            new_pos[idx, 1] = old_y

    return new_pos


def _find_vacant_region_centroids(occupied, cell_w, cell_h, grid_size):
    """
    Connected-component labeling on vacant cells.
    Returns the centroid (in canvas coords) of each connected vacant region.
    This matches CARDS Fig. 2 — the red/blue dots are region centroids, not
    individual cell centers.
    """ 
    visited = np.zeros_like(occupied, dtype=bool)
    centroids = []

    for r0 in range(grid_size):
        for c0 in range(grid_size):
            if occupied[r0, c0] or visited[r0, c0]:
                continue
            # BFS flood fill
            queue = [(r0, c0)]
            visited[r0, c0] = True
            region = []
            while queue:
                r, c = queue.pop()
                region.append((r, c))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        if not occupied[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            # Centroid of the connected region in canvas coordinates
            cx = np.mean([(c + 0.5) * cell_w for _, c in region])
            cy = np.mean([(r + 0.5) * cell_h for _, r in region])
            centroids.append([cx, cy])

    return centroids


# ---------------------------------------------------------------------------
# Main placer class
# ---------------------------------------------------------------------------

class GeneticPlacer:
    """
    Genetic Algorithm Macro Placer with CARDS-style Post-Processing.

    Phase 1 — Diverse initialization:
      Multiple legal starting placements (different greedy sort orders,
      edge-biased placement for large macros, legalized initial positions,
      SA-perturbed variants for diversity).

    Phase 2 — GA evolution:
      Tournament selection + position crossover + SA mutation.
      Fitness = HPWL (fast; full proxy cost used only for final ranking).
      Elitism preserves the best placements across generations.

    Phase 3 — CARDS post-processing:
      Vacant centroid movement on the best individual to reduce density.
      Final SA polish to tighten wirelength.
    """

    def __init__(
        self,
        seed: int = 42,
        pop_size: int = 30,
        n_generations: int = 40,
        elite_frac: float = 0.33,
        n_cards_passes: int = 3,
    ):
        self.seed = seed
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.elite_frac = elite_frac
        self.n_cards_passes = n_cards_passes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed)
        np.random.seed(self.seed)

        n = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        movable_mask = benchmark.get_movable_mask()[:n].numpy()
        movable_idx = np.where(movable_mask)[0].tolist()
        areas = sizes[:, 0] * sizes[:, 1]

        sep_x, sep_y = _make_sep(sizes)

        # Load connectivity graph
        plc = _load_plc(benchmark.name)
        if plc is not None:
            edges, weights = _extract_edges(benchmark, plc)
        else:
            edges = np.zeros((0, 2), dtype=np.int32)
            weights = np.zeros(0, dtype=np.float32)

        neighbors = _build_neighbors(n, edges)
        init_pos = benchmark.macro_positions[:n].numpy().copy().astype(np.float64)

        if len(movable_idx) == 0:
            full = benchmark.macro_positions.clone()
            return full

        # Phase 1: Initialize diverse legal population
        population = self._init_population(
            movable_idx, movable_mask, sizes, half_w, half_h,
            cw, ch, init_pos, areas, edges, weights, neighbors, sep_x, sep_y, n,
        )

        # Phase 2: GA evolution
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        # Temperature schedule: anneal over all generations
        T_start = max(cw, ch) * 0.12
        T_end = max(cw, ch) * 0.001
        # SA steps per new child: more at start (exploration), fewer at end (exploitation)
        steps_per_child = max(150, 2400 // max(self.n_generations, 1))

        for gen in range(self.n_generations):
            # Evaluate proxy fitness for all individuals
            fitness = [_proxy_fitness(p, edges, weights, sizes, n, cw, ch) for p in population]

            # Rank by fitness (lower = better)
            ranked = sorted(range(len(population)), key=lambda i: fitness[i])
            population = [population[i] for i in ranked]
            fitness = [fitness[i] for i in ranked]

            # Elites pass through unchanged
            next_gen = [p.copy() for p in population[:n_elite]]

            # Generation temperature (decays across generations)
            frac_gen = gen / max(self.n_generations - 1, 1)
            T_gen = T_start * (T_end / T_start) ** frac_gen

            # Rank-based selection weights (rank 1 most likely)
            n_pop = len(population)
            sel_weights = [1.0 / (i + 1) for i in range(n_pop)]

            while len(next_gen) < self.pop_size:
                # Tournament selection (rank-weighted)
                pa_idx = random.choices(range(n_pop), weights=sel_weights, k=1)[0]
                pb_idx = random.choices(range(n_pop), weights=sel_weights, k=1)[0]
                parent_a = population[pa_idx]
                parent_b = population[pb_idx]

                # Crossover (60% probability)
                if random.random() < 0.6 and pa_idx != pb_idx:
                    child = _crossover(
                        parent_a, parent_b, movable_idx,
                        half_w, half_h, cw, ch, sep_x, sep_y, n,
                    )
                else:
                    child = parent_a.copy()

                # SA mutation to refine the child
                child = _sa_evolve(
                    child, movable_idx, neighbors,
                    sizes, half_w, half_h, cw, ch, sep_x, sep_y, n,
                    edges, weights,
                    n_steps=steps_per_child,
                    T_start=T_gen * 2.0,
                    T_end=T_gen * 0.1,
                )
                next_gen.append(child)

            population = next_gen

        # Pick best individual by proxy fitness
        fitness = [_proxy_fitness(p, edges, weights, sizes, n, cw, ch) for p in population]
        best_pos = population[int(np.argmin(fitness))].copy()

        # Phase 3: CARDS vacant centroid movement (reduce density/congestion)
        # for _ in range(self.n_cards_passes):
        #     best_pos = _vacant_centroid_move(
        #         best_pos, movable_mask, sizes, half_w, half_h,
        #         cw, ch, sep_x, sep_y, n,
        #     )
        # for pass_i in range(self.n_cards_passes):
        #     prev_density = _fast_density_cost(best_pos, sizes, n, cw, ch)
        #     best_pos = _vacant_centroid_move(
        #         best_pos, movable_mask, sizes, half_w, half_h,
        #         cw, ch, sep_x, sep_y, n,
        #     )
        #     new_density = _fast_density_cost(best_pos, sizes, n, cw, ch)
        #     # Early stop if CARDS is making things worse
        #     if new_density > prev_density * 1.02:
        #         break

        # Final SA polish: tighten wirelength at low temperature
        best_pos = _sa_evolve(
            best_pos, movable_idx, neighbors,
            sizes, half_w, half_h, cw, ch, sep_x, sep_y, n,
            edges, weights,
            n_steps=1500,
            T_start=max(cw, ch) * 0.02,
            T_end=max(cw, ch) * 0.0001,
        )

        # Assemble full placement (hard macros updated, soft macros unchanged)
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n] = torch.tensor(best_pos, dtype=torch.float32)
        return full_pos

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------

    def _init_population(
        self, movable_idx, movable_mask, sizes, half_w, half_h,
        cw, ch, init_pos, areas, edges, weights, neighbors, sep_x, sep_y, n,
    ):
        """
        Build a diverse initial population of legal placements.

        Starting points:
          1. Greedy row placement with 4 different sort orders
          2. Edge-biased placement (large macros → canvas boundary)
          3. Legalized initial placement (minimum displacement from reference)
          4. SA-perturbed variants of the above for population diversity
        """
        pop = []

        # 1. Greedy placements with different sort orders
        sort_keys = [
            None,                                     # Tallest first (default)
            lambda i: -sizes[i, 0],                   # Widest first
            lambda i: -(sizes[i, 0] * sizes[i, 1]),   # Largest area first
            lambda i: sizes[i, 0] * sizes[i, 1],      # Smallest area first
        ]
        for sk in sort_keys:
            p = _greedy_row_place(movable_idx, sizes, half_w, half_h, cw, ch, init_pos, sk)
            pop.append(p)

        # 2. Edge-biased placement (large macros to boundary, then legalized)
        p = _edge_biased_place(movable_idx, sizes, half_w, half_h, cw, ch, init_pos, areas)
        p = _legalize(p, movable_mask, sizes, half_w, half_h, cw, ch, sep_x, sep_y)
        pop.append(p)

        # 3. Legalized initial reference placement
        p = _legalize(init_pos.copy(), movable_mask, sizes, half_w, half_h, cw, ch, sep_x, sep_y)
        pop.append(p)

        # 4. SA-evolved variants for diversity
        T0 = max(cw, ch) * 0.15
        T1 = max(cw, ch) * 0.005
        while len(pop) < self.pop_size:
            base = pop[random.randint(0, len(pop) - 1)]
            evolved = _sa_evolve(
                base.copy(), movable_idx, neighbors,
                sizes, half_w, half_h, cw, ch, sep_x, sep_y, n,
                edges, weights,
                n_steps=500, T_start=T0, T_end=T1,
            )
            pop.append(evolved)

        return pop[: self.pop_size]
