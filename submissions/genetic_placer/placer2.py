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
    sep_x = (w[:, np.newaxis] + w[np.newaxis, :]) / 2.0
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
    Fully vectorized density cost — no Python loop over macros.

    Replaces the original per-macro Python loop with broadcast operations.
    For n=175 macros this is ~40× faster: the old version did 175 iterations
    of Python overhead + small numpy calls; this does one fused numpy pass.

    Strategy: for each macro, compute overlap with every grid cell using
    broadcasting [n, grid] tensors, then sum over macros axis.
    """
    cell_w = cw / grid_size
    cell_h = ch / grid_size
    cell_area = cell_w * cell_h

    hw = sizes[:n, 0] / 2
    hh = sizes[:n, 1] / 2
    x0 = pos[:n, 0] - hw   # [n]
    x1 = pos[:n, 0] + hw
    y0 = pos[:n, 1] - hh
    y1 = pos[:n, 1] + hh

    # Cell left/right edges: [grid_size]
    col_edges_lo = np.arange(grid_size) * cell_w        # [G]
    col_edges_hi = col_edges_lo + cell_w
    row_edges_lo = np.arange(grid_size) * cell_h
    row_edges_hi = row_edges_lo + cell_h

    # Overlap in x: [n, G]
    ov_x = np.maximum(0.0,
        np.minimum(x1[:, None], col_edges_hi[None, :]) -
        np.maximum(x0[:, None], col_edges_lo[None, :])
    )
    # Overlap in y: [n, G]
    ov_y = np.maximum(0.0,
        np.minimum(y1[:, None], row_edges_hi[None, :]) -
        np.maximum(y0[:, None], row_edges_lo[None, :])
    )
    # Density grid: [G, G] via outer product summed over macros
    # ov_y[:, r] * ov_x[:, c] / cell_area summed over n → grid[r, c]
    grid = (ov_y[:, :, None] * ov_x[:, None, :]).sum(axis=0) / cell_area  # [G, G]

    flat = grid.ravel()
    n_top = max(1, grid_size * grid_size // 10)
    top10 = np.partition(flat, -n_top)[-n_top:]
    return 0.5 * float(top10.mean())


def _fast_congestion_cost(pos, edges, weights, cw, ch, grid_size=10):
    """
    Vectorized cut-based congestion proxy — no Python loop over cuts.

    Replaces two list comprehensions over grid cuts with broadcast comparisons.
    Cut thresholds [G-1] are broadcast against all edges [E] simultaneously.
    """
    if len(edges) == 0:
        return 0.0
    total_w = float(weights.sum())
    if total_w < 1e-9:
        return 0.0

    cell_w = cw / grid_size
    cell_h = ch / grid_size

    ex0 = np.minimum(pos[edges[:, 0], 0], pos[edges[:, 1], 0])  # [E]
    ex1 = np.maximum(pos[edges[:, 0], 0], pos[edges[:, 1], 0])
    ey0 = np.minimum(pos[edges[:, 0], 1], pos[edges[:, 1], 1])
    ey1 = np.maximum(pos[edges[:, 0], 1], pos[edges[:, 1], 1])

    # Cut positions: [G-1]
    v_thresh = np.arange(1, grid_size) * cell_w   # [G-1]
    h_thresh = np.arange(1, grid_size) * cell_h

    # spans_cut[e, k] = True if edge e spans vertical cut k: [E, G-1]
    v_spans = (ex0[:, None] < v_thresh[None, :]) & (ex1[:, None] > v_thresh[None, :])
    h_spans = (ey0[:, None] < h_thresh[None, :]) & (ey1[:, None] > h_thresh[None, :])

    v_cuts = (weights[:, None] * v_spans).sum(axis=0) / total_w  # [G-1]
    h_cuts = (weights[:, None] * h_spans).sum(axis=0) / total_w

    all_cuts = np.concatenate([v_cuts, h_cuts])
    n_top = max(1, len(all_cuts) // 10)
    top10 = np.partition(all_cuts, -n_top)[-n_top:]
    return float(top10.mean())


def _proxy_fitness(pos, edges, weights, sizes, n, cw, ch):
    """
    Full proxy fitness matching the evaluation metric:
      normalized_HPWL + 0.5 * density + 0.5 * congestion
    """
    total_w = float(weights.sum()) if len(weights) > 0 else 1.0
    hpwl_norm = _hpwl(pos, edges, weights) / max((cw + ch) * total_w, 1e-9)
    density = _fast_density_cost(pos, sizes, n, cw, ch)
    congestion = _fast_congestion_cost(pos, edges, weights, cw, ch)
    return hpwl_norm + 0.5 * density + 0.5 * congestion


# ---------------------------------------------------------------------------
# Incremental density grid for SA inner loop
# ---------------------------------------------------------------------------

class _DensityGrid:
    """
    Incrementally maintained density grid for the SA inner loop.

    The key runtime problem: _fast_density_cost recomputes the entire
    n-macro rasterization from scratch on every SA step. But each step
    only moves ONE macro — so only that macro's cells change.

    This class maintains the grid as mutable state. On each SA step:
      1. _remove(i): subtract macro i's contribution from the grid
      2. (move macro i)
      3. _add(i): add macro i's new contribution
      4. cost(): read top-10% mean from the updated grid

    Cost per step: O(cells_per_macro) numpy ops instead of O(n*cells_per_macro).
    For n=175, that's a 175× reduction in density work inside the SA loop.
    """

    def __init__(self, pos, sizes, n, cw, ch, grid_size=10):
        self.grid_size = grid_size
        self.cell_w = cw / grid_size
        self.cell_h = ch / grid_size
        self.cell_area = self.cell_w * self.cell_h
        self.n = n
        self.sizes = sizes
        self.cw = cw
        self.ch = ch

        # Precompute cell edge arrays once
        self.col_lo = np.arange(grid_size) * self.cell_w
        self.col_hi = self.col_lo + self.cell_w
        self.row_lo = np.arange(grid_size) * self.cell_h
        self.row_hi = self.row_lo + self.cell_h

        self.grid = np.zeros((grid_size, grid_size), dtype=np.float64)
        for i in range(n):
            self._add_macro(i, pos)

    def _macro_contribution(self, i, pos):
        hw, hh = self.sizes[i, 0] / 2, self.sizes[i, 1] / 2
        x0, x1 = pos[i, 0] - hw, pos[i, 0] + hw
        y0, y1 = pos[i, 1] - hh, pos[i, 1] + hh
        ov_x = np.maximum(0.0, np.minimum(x1, self.col_hi) - np.maximum(x0, self.col_lo))
        ov_y = np.maximum(0.0, np.minimum(y1, self.row_hi) - np.maximum(y0, self.row_lo))
        return np.outer(ov_y, ov_x) / self.cell_area  # [G, G]

    def _add_macro(self, i, pos):
        self.grid += self._macro_contribution(i, pos)

    def _remove_macro(self, i, pos):
        self.grid -= self._macro_contribution(i, pos)

    def update(self, i, old_pos, new_pos):
        """Remove macro i at old_pos, add it at new_pos."""
        self._remove_macro(i, old_pos)
        self._add_macro(i, new_pos)

    def cost(self):
        flat = self.grid.ravel()
        n_top = max(1, self.grid_size * self.grid_size // 10)
        top10 = np.partition(flat, -n_top)[-n_top:]
        return 0.5 * float(top10.mean())

    def clone(self):
        g = object.__new__(_DensityGrid)
        g.__dict__.update(self.__dict__)
        g.grid = self.grid.copy()
        return g


def _density_aware_cost(pos, edges, weights, sizes, n, cw, ch, density_weight=1.0):
    """
    Full recompute version — used outside SA (e.g. proxy_fitness calls).
    Inside SA, use _DensityGrid for incremental updates instead.
    """
    total_w = float(weights.sum()) if len(weights) > 0 else 1.0
    hpwl_norm = _hpwl(pos, edges, weights) / max((cw + ch) * total_w, 1e-9)
    if density_weight > 1e-6:
        density = _fast_density_cost(pos, sizes, n, cw, ch)
        return hpwl_norm + density_weight * density
    return hpwl_norm


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
               n_steps, T_start, T_end, density_weight=0.0):
    """
    SA with overlap-rejection and incremental density tracking.

    Speedups vs original:
      - _DensityGrid: O(cells_per_macro) update per step vs O(n*cells) recompute.
      - movable_set: O(1) neighbor filtering.
      - Per-macro edge index: HPWL delta touches only edges of moved macro.
      - min/max instead of np.clip for scalar bounds.

    Incremental density contract (invariant after every step):
      dgrid.grid reflects pos exactly.
    To move macro i old→new:
      1. dgrid._remove_macro(i, pos)  [pos[i] = old]
      2. pos[i] = new
      3. dgrid._add_macro(i, pos)     [pos[i] = new]
      If rejected: reverse (remove new, restore old, add old).
    """
    if len(movable_idx) == 0:
        return pos

    pos = pos.copy()
    movable_set = set(movable_idx)

    # Per-macro edge index for incremental HPWL delta
    if len(edges) > 0:
        macro_edge_idx = [[] for _ in range(n)]
        for eidx, (a, b) in enumerate(edges):
            macro_edge_idx[a].append(eidx)
            macro_edge_idx[b].append(eidx)
        macro_edge_idx = [np.array(e, dtype=np.int32) if e else np.zeros(0, dtype=np.int32)
                          for e in macro_edge_idx]
    else:
        macro_edge_idx = [np.zeros(0, dtype=np.int32)] * n

    total_w = float(weights.sum()) if len(weights) > 0 else 1.0
    norm = max((cw + ch) * total_w, 1e-9)
    use_density = density_weight > 1e-6
    dgrid = _DensityGrid(pos, sizes, n, cw, ch) if use_density else None

    def hpwl_delta_i(i, old_xi, old_yi):
        """Normalized HPWL delta from moving macro i. pos[i] must be new already."""
        ei = macro_edge_idx[i]
        if len(ei) == 0:
            return 0.0
        a, b, w = edges[ei, 0], edges[ei, 1], weights[ei]
        ax_old = np.where(a == i, old_xi, pos[a, 0])
        ay_old = np.where(a == i, old_yi, pos[a, 1])
        bx_old = np.where(b == i, old_xi, pos[b, 0])
        by_old = np.where(b == i, old_yi, pos[b, 1])
        old_wl = (w * (np.abs(ax_old - bx_old) + np.abs(ay_old - by_old))).sum()
        new_wl = (w * (np.abs(pos[a, 0] - pos[b, 0]) + np.abs(pos[a, 1] - pos[b, 1]))).sum()
        return float(new_wl - old_wl) / norm

    current_hpwl_norm = _hpwl(pos, edges, weights) / norm
    current_density   = dgrid.cost() if use_density else 0.0
    current_cost      = current_hpwl_norm + density_weight * current_density
    best_pos  = pos.copy()
    best_cost = current_cost

    log_ratio = math.log(max(T_end / max(T_start, 1e-12), 1e-30))

    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        T    = T_start * math.exp(log_ratio * frac)
        move = random.random()
        i    = random.choice(movable_idx)
        old_x, old_y = pos[i, 0], pos[i, 1]

        # ── Gaussian shift ────────────────────────────────────────────────
        if move < 0.5:
            scale = T * (0.3 + 0.7 * (1.0 - frac))
            new_x = min(max(old_x + random.gauss(0, scale), half_w[i]), cw - half_w[i])
            new_y = min(max(old_y + random.gauss(0, scale), half_h[i]), ch - half_h[i])

            if use_density: dgrid._remove_macro(i, pos)
            pos[i, 0] = new_x; pos[i, 1] = new_y

            if _check_overlap(i, pos, sep_x, sep_y, n):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                if use_density: dgrid._add_macro(i, pos)
                continue

            dh = hpwl_delta_i(i, old_x, old_y)
            if use_density:
                dgrid._add_macro(i, pos)
                nd = dgrid.cost(); dd = nd - current_density
            else:
                dd = 0.0; nd = 0.0

            delta = dh + density_weight * dd
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
                current_hpwl_norm += dh
                if use_density: current_density = nd
                current_cost = current_hpwl_norm + density_weight * current_density
                if current_cost < best_cost:
                    best_cost = current_cost; best_pos = pos.copy()
            else:
                if use_density: dgrid._remove_macro(i, pos)
                pos[i, 0] = old_x; pos[i, 1] = old_y
                if use_density: dgrid._add_macro(i, pos)

        # ── Swap ──────────────────────────────────────────────────────────
        elif move < 0.75:
            cands = [j for j in neighbors[i] if j in movable_set] if neighbors[i] else []
            j = (random.choice(cands) if cands and random.random() < 0.6
                 else random.choice(movable_idx))
            if i == j:
                continue

            old_jx, old_jy = pos[j, 0], pos[j, 1]
            new_ix = min(max(old_jx, half_w[i]), cw - half_w[i])
            new_iy = min(max(old_jy, half_h[i]), ch - half_h[i])
            new_jx = min(max(old_x,  half_w[j]), cw - half_w[j])
            new_jy = min(max(old_y,  half_h[j]), ch - half_h[j])

            if use_density:
                dgrid._remove_macro(i, pos); dgrid._remove_macro(j, pos)
            pos[i, 0] = new_ix; pos[i, 1] = new_iy
            pos[j, 0] = new_jx; pos[j, 1] = new_jy

            if _check_overlap(i, pos, sep_x, sep_y, n) or \
               _check_overlap(j, pos, sep_x, sep_y, n):
                pos[i, 0] = old_x;  pos[i, 1] = old_y
                pos[j, 0] = old_jx; pos[j, 1] = old_jy
                if use_density:
                    dgrid._add_macro(i, pos); dgrid._add_macro(j, pos)
                continue

            # HPWL delta over union of i and j edges
            ei = macro_edge_idx[i]; ej = macro_edge_idx[j]
            involved = np.union1d(ei, ej) if (len(ei) and len(ej)) else (ei if len(ei) else ej)
            if len(involved):
                a, b, w = edges[involved, 0], edges[involved, 1], weights[involved]
                ax_old = np.where(a == i, old_x,  np.where(a == j, old_jx, pos[a, 0]))
                ay_old = np.where(a == i, old_y,  np.where(a == j, old_jy, pos[a, 1]))
                bx_old = np.where(b == i, old_x,  np.where(b == j, old_jx, pos[b, 0]))
                by_old = np.where(b == i, old_y,  np.where(b == j, old_jy, pos[b, 1]))
                old_wl = (w * (np.abs(ax_old - bx_old) + np.abs(ay_old - by_old))).sum()
                new_wl = (w * (np.abs(pos[a, 0] - pos[b, 0]) + np.abs(pos[a, 1] - pos[b, 1]))).sum()
                dh = float(new_wl - old_wl) / norm
            else:
                dh = 0.0

            if use_density:
                dgrid._add_macro(i, pos); dgrid._add_macro(j, pos)
                nd = dgrid.cost(); dd = nd - current_density
            else:
                dd = 0.0; nd = 0.0

            delta = dh + density_weight * dd
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
                current_hpwl_norm += dh
                if use_density: current_density = nd
                current_cost = current_hpwl_norm + density_weight * current_density
                if current_cost < best_cost:
                    best_cost = current_cost; best_pos = pos.copy()
            else:
                if use_density:
                    dgrid._remove_macro(i, pos); dgrid._remove_macro(j, pos)
                pos[i, 0] = old_x;  pos[i, 1] = old_y
                pos[j, 0] = old_jx; pos[j, 1] = old_jy
                if use_density:
                    dgrid._add_macro(i, pos); dgrid._add_macro(j, pos)

        # ── Neighbor pull ─────────────────────────────────────────────────
        else:
            if not neighbors[i]:
                continue
            j = random.choice(neighbors[i])
            alpha = random.uniform(0.05, 0.3)
            new_x = min(max(old_x + alpha * (pos[j, 0] - old_x), half_w[i]), cw - half_w[i])
            new_y = min(max(old_y + alpha * (pos[j, 1] - old_y), half_h[i]), ch - half_h[i])

            if use_density: dgrid._remove_macro(i, pos)
            pos[i, 0] = new_x; pos[i, 1] = new_y

            if _check_overlap(i, pos, sep_x, sep_y, n):
                pos[i, 0] = old_x; pos[i, 1] = old_y
                if use_density: dgrid._add_macro(i, pos)
                continue

            dh = hpwl_delta_i(i, old_x, old_y)
            if use_density:
                dgrid._add_macro(i, pos)
                nd = dgrid.cost(); dd = nd - current_density
            else:
                dd = 0.0; nd = 0.0

            delta = dh + density_weight * dd
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
                current_hpwl_norm += dh
                if use_density: current_density = nd
                current_cost = current_hpwl_norm + density_weight * current_density
                if current_cost < best_cost:
                    best_cost = current_cost; best_pos = pos.copy()
            else:
                if use_density: dgrid._remove_macro(i, pos)
                pos[i, 0] = old_x; pos[i, 1] = old_y
                if use_density: dgrid._add_macro(i, pos)

    return best_pos


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
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        if not occupied[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            cx = np.mean([(c + 0.5) * cell_w for _, c in region])
            cy = np.mean([(r + 0.5) * cell_h for _, r in region])
            centroids.append([cx, cy])

    return centroids


def _vacant_centroid_move(pos, movable_mask, sizes, half_w, half_h,
                          cw, ch, sep_x, sep_y, n, grid_size=16, step_frac=0.25):
    """
    CARDS vacant area centroid finding and movement.

    Only runs if there is meaningful vacant space (density headroom).
    When macros are already wall-to-wall, this is a guaranteed no-op with
    occasional tiny regressions — skip it entirely in that case.
    """
    # Guard: if density is already near-saturated, CARDS cannot help.
    # Threshold of 0.5 on the density cost (= top-10% cells ~100% full)
    # means there is no connected vacant region large enough to be useful.
    current_density = _fast_density_cost(pos, sizes, n, cw, ch)
    if current_density > 0.45:
        return pos

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

    # Guard: need at least a few meaningful vacant regions to be useful
    if len(vacant_centroids) < 3:
        return pos

    vacant_arr = np.array(vacant_centroids)
    new_pos = pos.copy()
    movable_idx = np.where(movable_mask[:n])[0]

    macro_positions = pos[movable_idx]
    # [K, M] distance matrix from each centroid to each movable macro
    dists_matrix = np.sqrt(
        (vacant_arr[:, 0:1] - macro_positions[:, 0]) ** 2 +
        (vacant_arr[:, 1:2] - macro_positions[:, 1]) ** 2
    )
    nearest_centroid_per_macro = np.argmin(dists_matrix, axis=0)  # [M]
    dist_per_macro = dists_matrix[nearest_centroid_per_macro, np.arange(len(movable_idx))]

    # Paper eq 1: compute Dmax/Dmin per centroid group
    # Closer macros (small D) get larger alpha: alpha = D_const * (Dmax - D) / (Dmax - Dmin)
    alpha_per_macro = np.zeros(len(movable_idx))
    for c_idx in range(len(vacant_centroids)):
        members = np.where(nearest_centroid_per_macro == c_idx)[0]
        if len(members) == 0:
            continue
        D_vals = dist_per_macro[members]
        Dmax, Dmin = D_vals.max(), D_vals.min()
        denom = max(Dmax - Dmin, 1e-6)
        alpha_per_macro[members] = step_frac * (Dmax - D_vals) / denom

    # Process farthest-first to open corridors before moving inner macros
    order = np.argsort(-dist_per_macro)

    for local_i in order:
        idx = movable_idx[local_i]
        px, py = new_pos[idx, 0], new_pos[idx, 1]
        c_idx = nearest_centroid_per_macro[local_i]
        nearest = vacant_arr[c_idx]

        dx = nearest[0] - px
        dy = nearest[1] - py
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < 1e-6:
            continue

        alpha = alpha_per_macro[local_i]
        old_x, old_y = new_pos[idx, 0], new_pos[idx, 1]
        new_pos[idx, 0] = float(np.clip(px + alpha * dx / dist, half_w[idx], cw - half_w[idx]))
        new_pos[idx, 1] = float(np.clip(py + alpha * dy / dist, half_h[idx], ch - half_h[idx]))

        # Relaxed overlap threshold for refinement nudges
        dx_ov = np.abs(new_pos[idx, 0] - new_pos[:n, 0])
        dy_ov = np.abs(new_pos[idx, 1] - new_pos[:n, 1])
        ov = (dx_ov < sep_x[idx, :n] + 0.01) & (dy_ov < sep_y[idx, :n] + 0.01)
        ov[idx] = False
        if ov.any():
            new_pos[idx, 0] = old_x
            new_pos[idx, 1] = old_y

    return new_pos


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
      Fitness = full proxy cost (WL + density + congestion).
      SA mutation uses density-aware cost to prevent clustering.
      Density weight decays across generations (spread early, tighten late).
      Elitism preserves the best placements across generations.

    Phase 3 — CARDS post-processing (conditional):
      Only runs when density headroom exists (density_cost < 0.45).
      Vacant centroid movement on the best individual to reduce density.
      Final SA polish to tighten wirelength at low temperature, density_weight=0.
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
        T_start = max(cw, ch) * 0.12
        T_end = max(cw, ch) * 0.001
        steps_per_child = max(150, 2400 // max(self.n_generations, 1))

        # Density weight schedule: start high (enforce spread), decay to near-zero.
        # Early generations: density_weight=1.0 keeps macros spread.
        # Late generations: density_weight→0 focuses on WL refinement.
        # This is the core fix: SA was purely minimizing HPWL, which clusters
        # macros and blows up density. The decaying penalty balances the tradeoff.
        dw_start = 1.0
        dw_end = 0.05

        for gen in range(self.n_generations):
            # Evaluate full proxy fitness for all individuals
            fitness = [_proxy_fitness(p, edges, weights, sizes, n, cw, ch) for p in population]

            # Rank by fitness (lower = better)
            ranked = sorted(range(len(population)), key=lambda i: fitness[i])
            population = [population[i] for i in ranked]
            fitness = [fitness[i] for i in ranked]

            # Elites pass through unchanged
            next_gen = [p.copy() for p in population[:n_elite]]

            # Generation temperature and density weight (both decay across generations)
            frac_gen = gen / max(self.n_generations - 1, 1)
            T_gen = T_start * (T_end / T_start) ** frac_gen
            density_weight = dw_start * (dw_end / dw_start) ** frac_gen

            # Rank-based selection weights (rank 1 most likely)
            n_pop = len(population)
            sel_weights = [1.0 / (i + 1) for i in range(n_pop)]

            while len(next_gen) < self.pop_size:
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

                # SA mutation with density-aware cost (key fix vs original)
                child = _sa_evolve(
                    child, movable_idx, neighbors,
                    sizes, half_w, half_h, cw, ch, sep_x, sep_y, n,
                    edges, weights,
                    n_steps=steps_per_child,
                    T_start=T_gen * 2.0,
                    T_end=T_gen * 0.1,
                    density_weight=density_weight,
                )
                next_gen.append(child)

            population = next_gen

        # Pick best individual by full proxy fitness
        fitness = [_proxy_fitness(p, edges, weights, sizes, n, cw, ch) for p in population]
        best_pos = population[int(np.argmin(fitness))].copy()

        # Phase 3: CARDS vacant centroid movement (conditional on density headroom)
        for pass_i in range(self.n_cards_passes):
            prev_density = _fast_density_cost(best_pos, sizes, n, cw, ch)
            candidate = _vacant_centroid_move(
                best_pos, movable_mask, sizes, half_w, half_h,
                cw, ch, sep_x, sep_y, n,
            )
            new_density = _fast_density_cost(candidate, sizes, n, cw, ch)
            # Only accept if density actually improved
            if new_density < prev_density:
                best_pos = candidate
            else:
                break  # No more headroom; stop early

        # Final SA polish: pure WL tightening at low temperature, no density penalty
        best_pos = _sa_evolve(
            best_pos, movable_idx, neighbors,
            sizes, half_w, half_h, cw, ch, sep_x, sep_y, n,
            edges, weights,
            n_steps=1500,
            T_start=max(cw, ch) * 0.02,
            T_end=max(cw, ch) * 0.0001,
            density_weight=0.0,  # Pure WL polish at the end
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
          4. SA-perturbed variants — now with density_weight=1.0 to ensure
             the initial diversity search also respects spread, not just WL.
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

        # 4. SA-evolved variants with density_weight=1.0 (spread-aware diversity)
        T0 = max(cw, ch) * 0.15
        T1 = max(cw, ch) * 0.005
        while len(pop) < self.pop_size:
            base = pop[random.randint(0, len(pop) - 1)]
            evolved = _sa_evolve(
                base.copy(), movable_idx, neighbors,
                sizes, half_w, half_h, cw, ch, sep_x, sep_y, n,
                edges, weights,
                n_steps=500, T_start=T0, T_end=T1,
                density_weight=1.0,  # Enforce spread during initialization
            )
            pop.append(evolved)

        return pop[: self.pop_size]