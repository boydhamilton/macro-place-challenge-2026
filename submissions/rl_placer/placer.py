"""
RL Placer - Inference-only submission

Loads a pre-trained PPO policy (produced by train.py) and runs one greedy
episode to place macros, then legalizes the result.

Train offline first:
    uv run python submissions/rl_placer/train.py -b ibm01
    uv run python submissions/rl_placer/train.py --all

Then evaluate:
    uv run evaluate submissions/rl_placer/placer.py
    uv run evaluate submissions/rl_placer/placer.py --all
    uv run evaluate submissions/rl_placer/placer.py -b ibm03
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import PlacementEnv, PlacementPolicy, MODELS_DIR  # noqa: E402

from macro_place.benchmark import Benchmark

GAP = 0.05


class RLPlacer:
    """
    Loads a pre-trained PPO policy for the given benchmark and runs one
    greedy (argmax) inference episode, then legalizes the result.
    Falls back to greedy row packing if no trained model is found.
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        model_path = MODELS_DIR / f"{benchmark.name}.pt"

        if not model_path.exists():
            print(f"  [RL] No model found at {model_path}. Run train.py first.")
            print(f"  [RL] Falling back to greedy row placement.")
            return self._greedy_fallback(benchmark)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = PlacementPolicy().to(device)

        # Load model — supports both new dict format and legacy state_dict format
        data = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(data, dict) and "state_dict" in data:
            policy.load_state_dict(data["state_dict"])
            adjacency = data.get("adjacency", None)
        else:
            # Legacy: raw state_dict (no connectivity info)
            policy.load_state_dict(data)
            adjacency = None

        policy.eval()
        adj_str = f"{sum(len(a) for a in adjacency)} edges" if adjacency else "no adjacency"
        print(f"  [RL] Loaded {model_path} | {adj_str} | running greedy inference on {device}")

        env = PlacementEnv(benchmark, adjacency=adjacency)
        if not env.movable_idx:
            return benchmark.macro_positions.clone()

        state = env.reset()
        done = False
        while not done:
            canvas_np, macro_feat_np, mask_np, _ = state
            eff_mask = mask_np if not mask_np.all() else np.zeros_like(mask_np)

            canvas_t = torch.tensor(canvas_np, dtype=torch.float32, device=device).unsqueeze(0)
            macro_t = torch.tensor(macro_feat_np, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(eff_mask, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                logits, _ = policy(canvas_t, macro_t, mask_t)
                action = logits.argmax(dim=-1).item()

            state, _, done = env.step(action)

        placement = benchmark.macro_positions.clone()
        placement[:benchmark.num_hard_macros] = torch.tensor(
            env.placement[:benchmark.num_hard_macros], dtype=torch.float32
        )
        return self._legalize(placement, benchmark)

    # ── Legalization ───────────────────────────────────────────────────────────
    def _legalize(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """
        Two-pass legalization:
          1. Force-directed push (fast, preserves RL layout).
          2. Greedy spiral-search for any remaining overlaps (guaranteed legal).
        """
        pos = placement.clone().numpy().astype(np.float64)
        sizes = benchmark.macro_sizes.numpy()
        n = benchmark.num_hard_macros
        movable = benchmark.get_movable_mask()[:n].numpy()
        cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
        hw = sizes[:n, 0] / 2.0
        hh = sizes[:n, 1] / 2.0

        for i in range(n):
            if movable[i]:
                pos[i, 0] = np.clip(pos[i, 0], hw[i], cw - hw[i])
                pos[i, 1] = np.clip(pos[i, 1], hh[i], ch - hh[i])

        # Pass 1: force-directed (200 iterations)
        for iteration in range(200):
            any_overlap = False
            for i in range(n):
                if not movable[i]:
                    continue
                dx = np.abs(pos[i, 0] - pos[:n, 0])
                dy = np.abs(pos[i, 1] - pos[:n, 1])
                ovals = (dx < hw[i] + hw[:n] + GAP) & (dy < hh[i] + hh[:n] + GAP)
                ovals[i] = False
                if not ovals.any():
                    continue
                any_overlap = True
                oi = np.where(ovals)[0]
                push_x = pos[i, 0] - pos[oi, 0].mean()
                push_y = pos[i, 1] - pos[oi, 1].mean()
                d = math.sqrt(push_x ** 2 + push_y ** 2) + 1e-6
                step = max(sizes[i, 0], sizes[i, 1]) * 0.5
                pos[i, 0] = np.clip(pos[i, 0] + push_x / d * step, hw[i], cw - hw[i])
                pos[i, 1] = np.clip(pos[i, 1] + push_y / d * step, hh[i], ch - hh[i])
            if not any_overlap:
                print(f"  [RL] Force-directed legalized in {iteration + 1} iters.")
                result = placement.clone()
                result[:n] = torch.tensor(pos[:n], dtype=torch.float32)
                return result

        # Pass 2: greedy spiral-search
        order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
        committed = np.zeros(n, dtype=bool)
        for i in order:
            if not movable[i]:
                committed[i] = True
                continue
            if committed.any():
                dx = np.abs(pos[i, 0] - pos[:n, 0])
                dy = np.abs(pos[i, 1] - pos[:n, 1])
                if not ((dx < hw[i] + hw[:n] + GAP) & (dy < hh[i] + hh[:n] + GAP) & committed).any():
                    committed[i] = True
                    continue
            origin = pos[i].copy()
            step = max(sizes[i, 0], sizes[i, 1]) * 0.25
            best_pos = origin.copy()
            best_dist = float("inf")
            for r in range(1, 300):
                found = False
                for dxm in range(-r, r + 1):
                    for dym in range(-r, r + 1):
                        if abs(dxm) != r and abs(dym) != r:
                            continue
                        cx = np.clip(origin[0] + dxm * step, hw[i], cw - hw[i])
                        cy = np.clip(origin[1] + dym * step, hh[i], ch - hh[i])
                        if committed.any():
                            dx2 = np.abs(cx - pos[:n, 0])
                            dy2 = np.abs(cy - pos[:n, 1])
                            if ((dx2 < hw[i] + hw[:n] + GAP) & (dy2 < hh[i] + hh[:n] + GAP) & committed).any():
                                continue
                        d2 = (cx - origin[0]) ** 2 + (cy - origin[1]) ** 2
                        if d2 < best_dist:
                            best_dist, best_pos, found = d2, np.array([cx, cy]), True
                if found:
                    break
            pos[i] = best_pos
            committed[i] = True

        print("  [RL] Spiral legalization complete.")
        result = placement.clone()
        result[:n] = torch.tensor(pos[:n], dtype=torch.float32)
        return result

    # ── Greedy row fallback ────────────────────────────────────────────────────
    def _greedy_fallback(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        indices = torch.where(movable)[0].tolist()
        sizes = benchmark.macro_sizes
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        indices.sort(key=lambda i: -sizes[i, 1].item())
        gap = 0.001
        cursor_x = cursor_y = row_h = 0.0
        for idx in indices:
            w, h = sizes[idx, 0].item(), sizes[idx, 1].item()
            if cursor_x + w > cw:
                cursor_x = 0.0
                cursor_y += row_h + gap
                row_h = 0.0
            if cursor_y + h > ch:
                placement[idx, 0] = w / 2
                placement[idx, 1] = h / 2
                continue
            placement[idx, 0] = cursor_x + w / 2
            placement[idx, 1] = cursor_y + h / 2
            cursor_x += w + gap
            row_h = max(row_h, h)
        return placement
