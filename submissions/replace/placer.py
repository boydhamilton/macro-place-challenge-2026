"""
RePlAce Analytic Global Placer

Uses the ePlace/RePlAce Nesterov gradient descent method for global placement,
followed by legalization and SA polish.

Pipeline:
  1. Initialize: greedy row placement (guaranteed legal starting point)
  2. Global:     RePlAce analytic placement (WAWL + density, Nesterov method)
  3. Legalize:   spiral-search legalization (zero overlaps, min displacement)
  4. Polish:     SA refinement (pure WL tightening at low temperature)

Based on:
  RePlAce: Advancing Solution Quality and Routability Validation in Global Placement
  Cheng et al., IEEE TCAD 2019.

  ePlace: Electrostatics-Based Placement Using Fast Fourier Transform and Nesterov's Method
  Lu et al., ACM TODAES 2015.

Usage:
    uv run evaluate submissions/replace/placer.py
    uv run evaluate submissions/replace/placer.py --all
    uv run evaluate submissions/replace/placer.py -b ibm01
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Import helpers from sibling submission directories
_SUBMISSIONS = Path(__file__).parent.parent
sys.path.insert(0, str(_SUBMISSIONS))

from genetic_placer.replace import _run_replace
from genetic_placer.genetic_cards_placer import (
    _load_plc,
    _extract_edges,
    _build_neighbors,
    _make_sep,
    _greedy_row_place,
    _legalize,
    _sa_evolve,
)

from macro_place.benchmark import Benchmark


class ReplacePlacer:
    """
    RePlAce analytic global placer.

    Uses the ePlace/RePlAce Nesterov method for global placement,
    then legalizes to guarantee zero overlaps, then SA-polishes wirelength.
    """

    def __init__(
        self,
        seed: int = 42,
        n_iters: int = 300,
        n_bins: int = 32,
        target_density: float = 0.85,
        sa_polish_steps: int = 2000,
    ):
        self.seed = seed
        self.n_iters = n_iters
        self.n_bins = n_bins
        self.target_density = target_density
        self.sa_polish_steps = sa_polish_steps

    def place(self, benchmark: Benchmark) -> torch.Tensor:
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

        # Load net connectivity
        plc = _load_plc(benchmark.name)
        if plc is not None:
            edges, weights = _extract_edges(benchmark, plc)
        else:
            edges = np.zeros((0, 2), dtype=np.int32)
            weights = np.zeros(0, dtype=np.float32)

        neighbors = _build_neighbors(n, edges)
        init_pos = benchmark.macro_positions[:n].numpy().copy().astype(np.float64)

        if len(movable_idx) == 0:
            return benchmark.macro_positions.clone()

        # Step 1: Greedy row placement as legal starting point for RePlAce
        start_pos = _greedy_row_place(
            movable_idx, sizes, half_w, half_h, cw, ch, init_pos
        )

        # Step 2: RePlAce analytic global placement
        # Minimizes WAWL(x) + λ·density_penalty(x) via Nesterov/FISTA
        print(f"  RePlAce: {self.n_iters} iters, {self.n_bins}×{self.n_bins} bins, "
              f"target_density={self.target_density}")
        global_pos = _run_replace(
            movable_idx, movable_mask, sizes, half_w, half_h,
            cw, ch, start_pos, areas, edges, weights, n,
            n_iters=self.n_iters,
            n_bins=self.n_bins,
            target_density=self.target_density,
        )

        # Step 3: Legalize — RePlAce produces continuous positions; resolve overlaps
        print("  Legalizing...")
        legal_pos = _legalize(
            global_pos, movable_mask, sizes, half_w, half_h, cw, ch, sep_x, sep_y
        )

        # Step 4: SA polish — pure wirelength tightening at low temperature
        if self.sa_polish_steps > 0:
            print(f"  SA polish: {self.sa_polish_steps} steps...")
            legal_pos = _sa_evolve(
                legal_pos, movable_idx, neighbors,
                sizes, half_w, half_h, cw, ch, sep_x, sep_y, n,
                edges, weights,
                n_steps=self.sa_polish_steps,
                T_start=max(cw, ch) * 0.02,
                T_end=max(cw, ch) * 0.0001,
                density_weight=0.0,
            )

        # Assemble full placement tensor (hard macros updated, soft macros unchanged)
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n] = torch.tensor(legal_pos, dtype=torch.float32)
        return full_pos


if __name__ == "__main__":
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement

    print("=" * 60)
    print("RePlAce Analytic Placer")
    print("=" * 60)

    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )
    print(f"\nLoaded {benchmark.name}: {benchmark.num_hard_macros} hard macros, "
          f"canvas {benchmark.canvas_width:.1f}×{benchmark.canvas_height:.1f}")

    placer = ReplacePlacer()
    placement = placer.place(benchmark)

    is_valid, violations = validate_placement(placement, benchmark)
    print(f"\nValid: {is_valid}")
    if not is_valid:
        for v in violations[:5]:
            print(f"  {v}")

    costs = compute_proxy_cost(placement, benchmark, plc)
    print(f"\nProxy cost: {costs['proxy_cost']:.6f}")
    print(f"  Wirelength:  {costs['wirelength_cost']:.6f}")
    print(f"  Density:     {costs['density_cost']:.6f}")
    print(f"  Congestion:  {costs['congestion_cost']:.6f}")
    print(f"  Overlaps:    {costs['overlap_count']}")
