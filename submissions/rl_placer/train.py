"""
RL Placer - Offline Training Script

Trains a PPO policy for each benchmark and saves model weights to
submissions/rl_placer/models/{benchmark_name}.pt

Usage:
    uv run python submissions/rl_placer/train.py                    # train ibm01
    uv run python submissions/rl_placer/train.py -b ibm03
    uv run python submissions/rl_placer/train.py --all              # all 17 IBM benchmarks
    uv run python submissions/rl_placer/train.py --all --seconds 3600
"""

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from macro_place.benchmark import Benchmark

# ── Hyperparameters (shared with placer.py) ────────────────────────────────────
GRID_H = 16
GRID_W = 16
LR = 3e-4
GAMMA = 0.99
PPO_CLIP = 0.2
PPO_EPOCHS = 4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.02
MAX_GRAD_NORM = 0.5
ROLLOUT_EPISODES = 8
DEFAULT_TRAIN_SECONDS = 1800
# Per-step connectivity reward weight (relative to terminal spread/overlap reward)
CONNECTIVITY_STEP_WEIGHT = 1.0

MODELS_DIR = Path(__file__).parent / "models"

IBM_BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",
]


# ── Net connectivity extraction ────────────────────────────────────────────────
def _build_adjacency(plc, benchmark: Benchmark) -> list:
    """
    Build macro-to-macro adjacency list from plc net data.

    Iterates over plc.nets (driver_pin -> [sink_pins]) and finds which
    macros share nets, including both hard and soft macros.

    Returns adj[tensor_i] = list of tensor indices of connected macros.
    """
    # macro_name -> tensor_idx mapping
    macro_name_to_tensor = {}
    for tensor_i, plc_i in enumerate(benchmark.hard_macro_indices):
        name = plc.modules_w_pins[plc_i].get_name()
        macro_name_to_tensor[name] = tensor_i
    for tensor_i, plc_i in enumerate(benchmark.soft_macro_indices):
        name = plc.modules_w_pins[plc_i].get_name()
        macro_name_to_tensor[name] = benchmark.num_hard_macros + tensor_i

    # pin_name -> tensor_idx for fast lookup in nets dict
    pin_to_tensor = {}
    for macro_name, tensor_i in macro_name_to_tensor.items():
        for pin_name in plc.hard_macros_to_inpins.get(macro_name, []):
            pin_to_tensor[pin_name] = tensor_i
        for pin_name in plc.soft_macros_to_inpins.get(macro_name, []):
            pin_to_tensor[pin_name] = tensor_i

    # Build undirected adjacency from all nets
    n = benchmark.num_macros
    adj = [set() for _ in range(n)]
    for driver_pin_name, sink_pin_names in plc.nets.items():
        members = set()
        t = pin_to_tensor.get(driver_pin_name)
        if t is not None:
            members.add(t)
        for sink_name in sink_pin_names:
            t = pin_to_tensor.get(sink_name)
            if t is not None:
                members.add(t)
        members_list = list(members)
        for i_m, a in enumerate(members_list):
            for b in members_list[i_m + 1:]:
                adj[a].add(b)
                adj[b].add(a)

    return [list(s) for s in adj]


# ── Proxy reward (overlap + spread) ───────────────────────────────────────────
def compute_reward(positions: np.ndarray, sizes: np.ndarray, canvas_w: float, canvas_h: float) -> float:
    """
    Proxy reward using overlap penalty and density spread.

    Overlap: penalizes pairs of macros whose bounding boxes intersect.
    Spread:  rewards placements where macros are distributed across the canvas
             (low variance of grid-cell occupancy = even spread).

    Returns a value in roughly [-1, 0], where 0 is ideal (no overlaps, perfect spread).
    """
    n = len(positions)
    if n == 0:
        return 0.0

    hw = sizes[:, 0] / 2.0
    hh = sizes[:, 1] / 2.0

    # Overlap penalty: sum of normalized overlap areas for all pairs
    overlap_penalty = 0.0
    canvas_area = canvas_w * canvas_h
    for i in range(n):
        dx = np.maximum(0.0, hw[i] + hw - np.abs(positions[i, 0] - positions[:, 0]))
        dy = np.maximum(0.0, hh[i] + hh - np.abs(positions[i, 1] - positions[:, 1]))
        dx[i] = 0.0  # exclude self
        overlap_area = (dx * dy).sum()
        overlap_penalty += overlap_area / canvas_area

    # Normalize by number of pairs
    n_pairs = max(1, n * (n - 1) / 2)
    overlap_penalty /= n_pairs

    # Spread reward: measure how evenly macros cover the canvas grid cells.
    # Count macros per cell; low std = even spread = good.
    grid_counts = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    cell_w = canvas_w / GRID_W
    cell_h = canvas_h / GRID_H
    for i in range(n):
        c = int(np.clip(positions[i, 0] / cell_w, 0, GRID_W - 1))
        r = int(np.clip(positions[i, 1] / cell_h, 0, GRID_H - 1))
        grid_counts[r, c] += 1.0
    spread_penalty = grid_counts.std() / (n / (GRID_H * GRID_W) + 1e-6)
    spread_penalty = min(spread_penalty, 5.0) / 5.0  # normalize to [0,1]

    return -(0.7 * overlap_penalty + 0.3 * spread_penalty)


# ── Placement Environment ──────────────────────────────────────────────────────
class PlacementEnv:
    """
    Sequential macro placement environment for RL training.

    Places movable hard macros one at a time (largest area first).
    State:  [occupancy_grid, density_heatmap, connectivity_heatmap] (3 x GH x GW)
            + macro features (3,)
    Action: grid cell index in [GH * GW]
    Reward: per-step connectivity reward + sparse terminal reward
    """

    def __init__(self, benchmark: Benchmark, adjacency: list = None):
        self.b = benchmark
        self.GH = GRID_H
        self.GW = GRID_W
        self.cell_w = benchmark.canvas_width / GRID_W
        self.cell_h = benchmark.canvas_height / GRID_H
        self.canvas_w = float(benchmark.canvas_width)
        self.canvas_h = float(benchmark.canvas_height)

        movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        self.movable_idx = torch.where(movable_mask)[0].tolist()
        sizes = benchmark.macro_sizes.numpy()
        # Place largest macros first (harder to place, most impact on quality)
        self.movable_idx.sort(key=lambda i: -(sizes[i, 0] * sizes[i, 1]))

        self.sizes = sizes
        self.half_w = sizes[:, 0] / 2.0
        self.half_h = sizes[:, 1] / 2.0

        # Precomputed grid cell centers
        cols = np.arange(GRID_W)
        rows = np.arange(GRID_H)
        self.cx_grid = ((cols + 0.5) * self.cell_w)[None, :].repeat(GRID_H, axis=0)
        self.cy_grid = ((rows + 0.5) * self.cell_h)[:, None].repeat(GRID_W, axis=1)

        # Macro-to-macro adjacency: adj[tensor_i] = [tensor_j, ...]
        self.adj = adjacency  # None if no connectivity info

        self.placement = None
        self.placed = None
        self.placed_list = None
        self.occupancy = None
        self.density = None
        self.step_idx = 0

    def reset(self):
        self.placement = self.b.macro_positions.clone().numpy().astype(np.float64)
        self.placed = set()
        self.placed_list = []
        self.occupancy = np.zeros((self.GH, self.GW), dtype=np.float32)
        self.density = np.zeros((self.GH, self.GW), dtype=np.float32)
        self.step_idx = 0
        # Mark fixed macros
        for i in range(self.b.num_hard_macros):
            if self.b.macro_fixed[i].item():
                self._mark_occ(i, self.placement[i, 0], self.placement[i, 1])
                self.placed.add(i)
                self.placed_list.append(i)
        return self._state()

    def _mark_occ(self, idx, x, y):
        hw, hh = self.half_w[idx], self.half_h[idx]
        c0 = max(0, int((x - hw) / self.cell_w))
        c1 = min(self.GW - 1, int((x + hw) / self.cell_w))
        r0 = max(0, int((y - hh) / self.cell_h))
        r1 = min(self.GH - 1, int((y + hh) / self.cell_h))
        self.occupancy[r0:r1 + 1, c0:c1 + 1] = min(
            1.0, self.occupancy[r0:r1 + 1, c0:c1 + 1].max() + 0.5
        )
        # Track area density for heatmap channel
        area = self.sizes[idx, 0] * self.sizes[idx, 1]
        cell_area = self.cell_w * self.cell_h
        self.density[r0:r1 + 1, c0:c1 + 1] += (area / max(cell_area, 1e-6)) / (
            (c1 - c0 + 1) * (r1 - r0 + 1)
        )

    def invalid_mask(self, idx) -> np.ndarray:
        """[GH*GW] bool: True = invalid action for macro idx."""
        hw, hh = self.half_w[idx], self.half_h[idx]
        oob = (
            (self.cx_grid - hw < 0) | (self.cx_grid + hw > self.canvas_w) |
            (self.cy_grid - hh < 0) | (self.cy_grid + hh > self.canvas_h)
        )
        overlap = np.zeros((self.GH, self.GW), dtype=bool)
        if self.placed_list:
            pl = np.array(self.placed_list)
            px = self.placement[pl, 0]
            py = self.placement[pl, 1]
            sx = hw + self.half_w[pl]
            sy = hh + self.half_h[pl]
            GAP = 0.05
            dx = np.abs(self.cx_grid[:, :, None] - px[None, None, :])
            dy = np.abs(self.cy_grid[:, :, None] - py[None, None, :])
            overlap = ((dx < sx[None, None, :] + GAP) & (dy < sy[None, None, :] + GAP)).any(axis=2)
        return (oob | overlap).reshape(-1)

    def _density_heatmap(self) -> np.ndarray:
        """Normalized area density across grid cells."""
        d = self.density.copy()
        mx = d.max()
        if mx > 0:
            d /= mx
        return d

    def _connectivity_heatmap(self, idx) -> np.ndarray:
        """
        [GH, GW] heatmap: cells where net-connected macros of *idx* are placed.

        Each placed neighbor increments its grid cell. Normalized to [0, 1].
        Returns zeros if no adjacency info or no placed neighbors.
        """
        hmap = np.zeros((self.GH, self.GW), dtype=np.float32)
        if self.adj is None:
            return hmap
        for j in self.adj[idx]:
            if j in self.placed:
                x, y = self.placement[j, 0], self.placement[j, 1]
                c = int(np.clip(x / self.cell_w, 0, self.GW - 1))
                r = int(np.clip(y / self.cell_h, 0, self.GH - 1))
                hmap[r, c] += 1.0
        mx = hmap.max()
        if mx > 0:
            hmap /= mx
        return hmap

    def _state(self):
        idx = self.movable_idx[min(self.step_idx, len(self.movable_idx) - 1)]
        canvas = np.stack([
            self.occupancy,
            self._density_heatmap(),
            self._connectivity_heatmap(idx),
        ], axis=0)  # [3, GH, GW]
        w, h = self.sizes[idx, 0], self.sizes[idx, 1]
        macro_feat = np.array([
            w / self.canvas_w,
            h / self.canvas_h,
            (w * h) / (self.canvas_w * self.canvas_h),
        ], dtype=np.float32)
        mask = self.invalid_mask(idx)
        return canvas, macro_feat, mask, idx

    def step(self, action: int):
        idx = self.movable_idx[self.step_idx]
        r, c = action // self.GW, action % self.GW
        x = np.clip((c + 0.5) * self.cell_w, self.half_w[idx], self.canvas_w - self.half_w[idx])
        y = np.clip((r + 0.5) * self.cell_h, self.half_h[idx], self.canvas_h - self.half_h[idx])
        self.placement[idx, 0] = x
        self.placement[idx, 1] = y
        self._mark_occ(idx, x, y)
        self.placed.add(idx)
        self.placed_list.append(idx)
        self.step_idx += 1
        done = self.step_idx >= len(self.movable_idx)

        # Per-step connectivity reward: penalize mean HPWL to already-placed neighbors.
        # Normalized so the sum over all steps is in [-1, 0] range.
        reward = 0.0
        if self.adj is not None:
            placed_neighbors = [j for j in self.adj[idx] if j in self.placed and j != idx]
            if placed_neighbors:
                total_hpwl = sum(
                    abs(x - self.placement[j, 0]) + abs(y - self.placement[j, 1])
                    for j in placed_neighbors
                )
                mean_hpwl = total_hpwl / len(placed_neighbors)
                # Normalize: taxi-cab canvas diagonal, divide by n_movable for per-step scale
                reward = (
                    -CONNECTIVITY_STEP_WEIGHT
                    * mean_hpwl
                    / (self.canvas_w + self.canvas_h)
                    / max(1, len(self.movable_idx))
                )

        if done:
            pl = np.array(self.placed_list)
            terminal_reward = compute_reward(
                self.placement[pl],
                self.sizes[pl],
                self.canvas_w,
                self.canvas_h,
            )
            reward += terminal_reward

        return self._state(), reward, done


# ── Policy Network ─────────────────────────────────────────────────────────────
class PlacementPolicy(nn.Module):
    """
    CNN+MLP policy over a discretized placement grid.

    Input:  canvas [B, 3, GH, GW]  (occupancy, density, connectivity)
            + macro_feat [B, 3]
    Output: action logits [B, GH*GW]  +  value scalar [B]
    """

    def __init__(self):
        super().__init__()
        n_actions = GRID_H * GRID_W
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        cnn_flat = 64 * GRID_H * GRID_W
        self.macro_enc = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(cnn_flat + 64, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def forward(self, canvas, macro_feat, inv_mask=None):
        h = self.trunk(torch.cat([self.cnn(canvas).flatten(1), self.macro_enc(macro_feat)], dim=-1))
        logits = self.policy_head(h)
        if inv_mask is not None:
            logits = logits.masked_fill(inv_mask, float("-inf"))
            all_masked = (logits == float("-inf")).all(dim=-1, keepdim=True)
            if all_masked.any():
                logits = torch.where(all_masked.expand_as(logits), torch.zeros_like(logits), logits)
        return logits, self.value_head(h).squeeze(-1)


# ── PPO Training ───────────────────────────────────────────────────────────────
def _compute_returns(transitions: list, gamma: float) -> list:
    G = 0.0
    returns = []
    for t in reversed(transitions):
        G = t["reward"] + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


def _ppo_update(policy, optimizer, device, buf: dict):
    canvas_t = torch.tensor(np.array(buf["canvas"]), dtype=torch.float32, device=device)
    macro_t = torch.tensor(np.array(buf["macro_feat"]), dtype=torch.float32, device=device)
    mask_t = torch.tensor(np.array(buf["masks"]), dtype=torch.bool, device=device)
    actions_t = torch.tensor(buf["actions"], dtype=torch.long, device=device)
    old_lp_t = torch.tensor(buf["log_probs"], dtype=torch.float32, device=device)
    returns_t = torch.tensor(buf["returns"], dtype=torch.float32, device=device)
    values_t = torch.tensor(buf["values"], dtype=torch.float32, device=device)

    adv = returns_t - values_t
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = len(buf["actions"])
    batch_size = min(256, N)
    for _ in range(PPO_EPOCHS):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            logits, values = policy(canvas_t[idx], macro_t[idx], mask_t[idx])
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(actions_t[idx])
            ratio = (new_lp - old_lp_t[idx]).exp()
            a = adv[idx]
            surr = torch.min(ratio * a, torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * a)
            loss = (-surr.mean()
                    + VALUE_COEF * F.mse_loss(values, returns_t[idx])
                    - ENTROPY_COEF * dist.entropy().mean())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()


def train(benchmark: Benchmark, plc=None, train_seconds: int = DEFAULT_TRAIN_SECONDS, seed: int = 42) -> tuple:
    """Train a PPO policy on *benchmark* and return (policy, adjacency)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build macro-to-macro adjacency from net data (if plc available)
    adjacency = None
    if plc is not None:
        print(f"  Building net adjacency from {int(plc.net_cnt)} nets...")
        adjacency = _build_adjacency(plc, benchmark)
        n_edges = sum(len(a) for a in adjacency)
        print(f"  Adjacency: {n_edges} directed edges across {benchmark.num_macros} macros")

    env = PlacementEnv(benchmark, adjacency=adjacency)

    if not env.movable_idx:
        return PlacementPolicy().to(device), adjacency

    policy = PlacementPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    best_reward = float("-inf")
    best_state_dict = {k: v.clone() for k, v in policy.state_dict().items()}

    buf = {"canvas": [], "macro_feat": [], "masks": [], "actions": [], "log_probs": [], "returns": [], "values": []}

    start = time.time()
    episode = 0

    conn_str = "with connectivity" if adjacency else "no connectivity"
    print(f"  Training {benchmark.name} | {len(env.movable_idx)} movable macros | "
          f"grid {GRID_H}×{GRID_W} | {train_seconds}s | {device} | {conn_str}")

    while (time.time() - start) < train_seconds:
        state = env.reset()
        transitions = []
        done = False

        while not done:
            canvas_np, macro_feat_np, mask_np, macro_idx = state
            eff_mask = mask_np if not mask_np.all() else np.zeros_like(mask_np)

            canvas_t = torch.tensor(canvas_np, dtype=torch.float32, device=device).unsqueeze(0)
            macro_t = torch.tensor(macro_feat_np, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(eff_mask, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                logits, value = policy(canvas_t, macro_t, mask_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, reward, done = env.step(action.item())
            transitions.append({
                "canvas": canvas_np, "macro_feat": macro_feat_np, "mask": eff_mask,
                "action": action.item(), "log_prob": log_prob.item(),
                "reward": reward, "value": value.item(),
            })
            state = next_state

        episode += 1
        returns = _compute_returns(transitions, GAMMA)
        # Use total episode return for best-model selection
        episode_return = sum(t["reward"] for t in transitions)

        if episode_return > best_reward:
            best_reward = episode_return
            print(f"    New best return: {best_reward:.6f} at episode {episode}")
            best_state_dict = {k: v.clone() for k, v in policy.state_dict().items()}

        for i, t in enumerate(transitions):
            buf["canvas"].append(t["canvas"])
            buf["macro_feat"].append(t["macro_feat"])
            buf["masks"].append(t["mask"])
            buf["actions"].append(t["action"])
            buf["log_probs"].append(t["log_prob"])
            buf["returns"].append(returns[i])
            buf["values"].append(t["value"])

        if episode % ROLLOUT_EPISODES == 0 and len(buf["actions"]) >= 32:
            _ppo_update(policy, optimizer, device, buf)
            for v in buf.values():
                v.clear()

        if episode % 50 == 0:
            elapsed = time.time() - start
            print(f"    ep={episode:4d}  best_return={best_reward:.6f}  {elapsed:.0f}s/{train_seconds}s")

    print(f"  Done. Episodes={episode}  best_return={best_reward:.4f}")
    policy.load_state_dict(best_state_dict)
    return policy, adjacency


def save_model(policy: PlacementPolicy, benchmark_name: str, adjacency=None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{benchmark_name}.pt"
    torch.save({
        "state_dict": policy.state_dict(),
        "adjacency": adjacency,
    }, path)
    print(f"  Saved → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train RL placement policy offline.")
    parser.add_argument("-b", "--benchmark", default="ibm01",
                        help="Single benchmark name (default: ibm01)")
    parser.add_argument("--all", action="store_true", help="Train all 17 IBM benchmarks")
    parser.add_argument("--seconds", type=int, default=DEFAULT_TRAIN_SECONDS,
                        help=f"Training time per benchmark in seconds (default: {DEFAULT_TRAIN_SECONDS})")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from macro_place.loader import load_benchmark_from_dir
    testcase_root = Path("external/MacroPlacement/Testcases/ICCAD04")

    benchmarks = IBM_BENCHMARKS if args.all else [args.benchmark]
    for name in benchmarks:
        print(f"\n{'='*60}")
        print(f"Benchmark: {name}")
        benchmark, plc = load_benchmark_from_dir(str(testcase_root / name))
        policy, adjacency = train(benchmark, plc=plc, train_seconds=args.seconds, seed=args.seed)
        save_model(policy, name, adjacency=adjacency)


if __name__ == "__main__":
    main()
