"""
RePlAce-inspired Macro Placer

Implements the five core components from the RePlAce/ePlace papers:
1. Per-net LSE HPWL wirelength (weighted-average model)
2. FFT-based Poisson electrostatic density (long-range spreading force)
3. Per-bin local density penalty (nu_j weighting)
4. Nesterov momentum with preconditioning (net-degree Jacobi preconditioner)
5. Adaptive grid resolution based on design size
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Net connectivity loading (same pattern as genetic placer)
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


def _extract_nets(benchmark, plc):
    """
    Extract hyperedges (per-net macro lists) from netlist.

    Returns:
        all_nodes: [total_pins] LongTensor - macro index per pin
        net_ids:   [total_pins] LongTensor - which net each pin belongs to
        weights:   [num_nets]   FloatTensor - net weight (1 / (n_pins - 1))
        n_nets:    int
    """
    # Map hard macro name -> benchmark tensor index [0, num_hard)
    name_to_bidx = {}
    for bidx, plc_idx in enumerate(benchmark.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[plc_idx].get_name()] = bidx

    net_node_lists = []
    net_weight_list = []

    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + list(sinks):
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) >= 2:
            macro_list = sorted(macros)
            net_node_lists.append(macro_list)
            # Standard net weight: 1/(pins-1) matches HPWL scaling
            net_weight_list.append(1.0 / (len(macro_list) - 1))

    if not net_node_lists:
        return (torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0),
                0)

    n_nets = len(net_node_lists)
    net_lens = [len(nl) for nl in net_node_lists]

    all_nodes = torch.tensor(
        [node for nl in net_node_lists for node in nl], dtype=torch.long
    )
    net_ids = torch.repeat_interleave(
        torch.arange(n_nets, dtype=torch.long),
        torch.tensor(net_lens, dtype=torch.long)
    )
    weights = torch.tensor(net_weight_list, dtype=torch.float32)

    return all_nodes, net_ids, weights, n_nets


# ---------------------------------------------------------------------------
# Legalization (unchanged from original)
# ---------------------------------------------------------------------------

def _make_sep(sizes):
    w = sizes[:, 0]
    h = sizes[:, 1]
    sep_x = (w[:, np.newaxis] + w[np.newaxis, :]) / 2.0
    sep_y = (h[:, np.newaxis] + h[np.newaxis, :]) / 2.0
    return sep_x, sep_y


def _legalize(pos, movable, sizes, half_w, half_h, cw, ch, sep_x, sep_y):
    n = len(pos)
    order = sorted(range(n), key=lambda i: -(sizes[i, 0] * sizes[i, 1]))
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        if placed.any():
            p = placed.copy()
            p[idx] = False
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            if not ((dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & p).any():
                placed[idx] = True
                continue

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
# Differentiable WL / density losses
# ---------------------------------------------------------------------------

def _scatter_logsumexp(src, index, num_segments):
    """Numerically stable scatter log-sum-exp (PyTorch 2.x)."""
    if src.numel() == 0:
        return torch.zeros(num_segments, device=src.device, dtype=src.dtype)
    max_val = torch.full((num_segments,), float('-inf'),
                         device=src.device, dtype=src.dtype)
    max_val.scatter_reduce_(0, index, src, reduce='amax', include_self=True)
    max_val = max_val.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    exp_shifted = torch.exp(src - max_val[index])
    sum_exp = torch.zeros(num_segments, device=src.device, dtype=src.dtype)
    sum_exp.scatter_add_(0, index, exp_shifted)
    return max_val + torch.log(sum_exp.clamp(min=1e-30))


def _wl_loss(pos, all_nodes, net_ids, n_nets, net_weights, gamma):
    """Per-net LSE HPWL approximation."""
    if n_nets == 0:
        return pos.sum() * 0.0  # keeps gradient graph alive

    pin_xy = pos[all_nodes]  # [total_pins, 2]
    px, py = pin_xy[:, 0], pin_xy[:, 1]

    lse_xp = gamma * _scatter_logsumexp( px / gamma, net_ids, n_nets)
    lse_xn = gamma * _scatter_logsumexp(-px / gamma, net_ids, n_nets)
    lse_yp = gamma * _scatter_logsumexp( py / gamma, net_ids, n_nets)
    lse_yn = gamma * _scatter_logsumexp(-py / gamma, net_ids, n_nets)

    return (net_weights * (lse_xp + lse_xn + lse_yp + lse_yn)).sum()


def _compute_density_map(pos, sizes, grid_res, bw, bh, device, chunk_size=128):
    """
    Project macro areas onto grid using the C1-continuous ePlace-MS bell function.
    Returns [grid_res, grid_res] density map.

    Processes macros in chunks to avoid materialising an [n, G, G] tensor.
    The bell support is [0, 1) in normalised distance u = dx / (w/2 + bw),
    so a macro only affects bins within one bell-width of its centre.
    """
    bx = (torch.arange(grid_res, device=device, dtype=pos.dtype) + 0.5) * bw
    by = (torch.arange(grid_res, device=device, dtype=pos.dtype) + 0.5) * bh

    density_map = None
    n = pos.shape[0]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        p = pos[start:end]     # [c, 2]
        s = sizes[start:end]   # [c, 2]

        dx = torch.abs(p[:, 0].unsqueeze(1) - bx.unsqueeze(0))  # [c, G]
        dy = torch.abs(p[:, 1].unsqueeze(1) - by.unsqueeze(0))  # [c, G]

        # a = macro_half_size + bin_size  (bell reaches zero at u=1)
        ax = (s[:, 0] / 2 + bw).unsqueeze(1).clamp(min=1e-6)  # [c, 1]
        ay = (s[:, 1] / 2 + bh).unsqueeze(1).clamp(min=1e-6)

        ux = dx / ax   # [c, G]
        uy = dy / ay

        # C1-continuous quadratic spline (ePlace-MS bell function):
        #   u < 0.5 : 1 - 2u²          (value=1 at u=0, value=0.5 at u=0.5)
        #   0.5≤u<1 : 2(1-u)²           (value=0.5 at u=0.5, value=0 at u=1)
        #   u ≥ 1   : 0
        # Derivative at u=0.5 from both sides = -2 → C1 continuous.
        def _bell(u):
            inner = 1.0 - 2.0 * u.pow(2)
            outer = 2.0 * (1.0 - u).pow(2)
            val = torch.where(u < 0.5, inner, outer)
            return torch.where(u < 1.0, val, torch.zeros_like(u))

        

        px  = _bell(ux)   # [c, G]
        py_ = _bell(uy)   # [c, G]

        macro_area = s[:, 0] * s[:, 1]
        scale = macro_area / (bw * bh)                          # [c]
        chunk_map = (scale.unsqueeze(1).unsqueeze(1)
                     * py_.unsqueeze(2) * px.unsqueeze(1)).sum(0)

        density_map = chunk_map if density_map is None else density_map + chunk_map

    if density_map is None:
        density_map = torch.zeros(grid_res, grid_res, device=device)

    return density_map


def _poisson_dct(excess, bw, bh, device):
    """
    Solve ∇²ψ = excess with Neumann (zero-flux) boundary conditions.

    Implementation: extend excess to 2N×2M by even reflection, then use
    the standard periodic FFT.  The periodic eigenvalues on the 2N×2M
    lattice are identical to the Neumann eigenvalues on N×M, so the top-
    left N×M quadrant of the IFFT result is the correct DCT-II solution.
    This avoids wrap-around repulsion from the opposite canvas edge that
    the plain periodic FFT would introduce.
    """
    N, M = excess.shape

    # 2D even reflection: [[A, flip_col(A)], [flip_row(A), flip_both(A)]]
    top = torch.cat([excess, excess.flip(1)], dim=1)     # [N, 2M]
    ext = torch.cat([top,    top.flip(0)],   dim=0)      # [2N, 2M]

    rho_hat = torch.fft.rfft2(ext)                       # [2N, M+1]

    m = torch.arange(2 * N,  device=device, dtype=torch.float32)
    n = torch.arange(M + 1,  device=device, dtype=torch.float32)

    # Eigenvalues of periodic Laplacian on 2N×2M  ≡  Neumann on N×M
    # λ_x[m] = (2cos(π m / N) − 2) / bw²
    eig_x = (2.0 * torch.cos(torch.pi * m / N) - 2.0) / (bw ** 2)
    eig_y = (2.0 * torch.cos(torch.pi * n / M) - 2.0) / (bh ** 2)
    eig = eig_x.unsqueeze(1) + eig_y.unsqueeze(0)       # [2N, M+1]
    eig[0, 0] = -1.0                                     # prevent ÷0; DC zeroed below

    psi_hat = -rho_hat / eig
    psi_hat[0, 0] = 0.0                                  # arbitrary DC constant

    psi_ext = torch.fft.irfft2(psi_hat, s=(2 * N, 2 * M))
    return psi_ext[:N, :M]


def _density_loss(pos, sizes, grid_res, bw, bh, cw, ch, device):
    """
    Electrostatic density loss with per-bin local penalty (νbelj weighting).

    Loss = 0.5 · Σ_j  ν_j · excess_j · ψ_j · bin_area

    ν_j (the RePlAce-ld per-bin weight) is computed without gradient so it
    acts as a spatially-varying multiplier: over-dense bins (ρ_j > ρ_target)
    get full weight; under-filled bins get a proportionally reduced weight.
    This concentrates spreading force where it is actually needed, preventing
    unnecessary wirelength sacrifice in already-spread regions.
    """
    density_map = _compute_density_map(pos, sizes, grid_res, bw, bh, device)
    target = (sizes[:, 0] * sizes[:, 1]).sum() / (cw * ch)
    excess = density_map - target

    with torch.no_grad():
        nu = (density_map / target.clamp(min=1e-8)).clamp(0.0, 2.0)  # [G, G]

    psi = _poisson_dct(excess, bw, bh, device)
    return 0.5 * (nu * excess * psi).sum() * bw * bh



# ---------------------------------------------------------------------------
# Main placer
# ---------------------------------------------------------------------------

class ImprovedReplacePlacer:
    def __init__(
        self,
        seed: int = 42,
        n_iters: int = 800,
        learning_rate: float = None,   # None = auto (0.5% of max canvas dim)
        grid_res: int = 64,
        gamma_scale: float = 0.01,     # gamma = gamma_scale * max(cw, ch)
        momentum: float = 0.9,
    ):
        self.seed = seed
        self.n_iters = n_iters
        self.lr = learning_rate
        self.grid_res = grid_res
        self.gamma_scale = gamma_scale
        self.momentum = momentum

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n_hard].to(device)   # hard macros only for density
        cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
        fixed = benchmark.macro_fixed[:n_hard].to(device)

        # --- Load net connectivity ---
        plc = _load_plc(benchmark.name)
        if plc is not None:
            all_nodes, net_ids, net_weights, n_nets = _extract_nets(benchmark, plc)
            all_nodes = all_nodes.to(device)
            net_ids = net_ids.to(device)
            net_weights = net_weights.to(device)
        else:
            all_nodes = torch.zeros(0, dtype=torch.long, device=device)
            net_ids = torch.zeros(0, dtype=torch.long, device=device)
            net_weights = torch.zeros(0, device=device)
            n_nets = 0

        # --- Preconditioning: Jacobi (sum of net weights per macro) ---
        precond = torch.ones(n_hard, device=device)
        if n_nets > 0:
            precond.scatter_add_(0, all_nodes, net_weights[net_ids])
        precond.clamp_(min=1.0)

        # --- Initialize positions (center + small noise) ---
        pos = torch.full((n_hard, 2), 0.5, device=device)
        pos[:, 0] *= cw
        pos[:, 1] *= ch
        pos += torch.randn_like(pos) * (max(cw, ch) * 0.005)
        # Restore fixed macros to their original positions
        if fixed.any():
            pos[fixed] = benchmark.macro_positions[:n_hard][fixed].to(device)

        gamma = max(cw, ch) * self.gamma_scale
        init_lr = self.lr if self.lr is not None else max(cw, ch) * 0.005
        lr = init_lr
        mu = self.momentum
        grid_res = self.grid_res
        bw = cw / grid_res
        bh = ch / grid_res

        density_weight = 1e-5
        v = torch.zeros_like(pos)

        # Lipschitz adaptive step-size state
        prev_grad     = None
        prev_pos_look = None
        lr_min = init_lr * 0.001
        lr_max = init_lr * 50.0

        # HPWL-plateau lambda ramping state (RePlAce Section IV-A)
        wl_history: list[float] = []
        # PLATEAU_WINDOW   = 10          # look back this many iters
        # PLATEAU_THRESH   = 0.005       # < 0.5 % improvement → plateau
        # LAMBDA_RAMP      = 1.15        # multiply density_weight by this on plateau
        # LAMBDA_MAX       = 1.0         # cap to keep WL competitive

        PLATEAU_WINDOW   = 20         # longer window, less trigger-happy
        PLATEAU_THRESH   = 0.01      # only ramp on very flat plateaus
        LAMBDA_RAMP      = 1.15       # gentler ramp (was 1.15)
        LAMBDA_MAX       = 0.01       # ← cap ~1000x lower; density is a penalty not the objective

        fixed_pos = benchmark.macro_positions[:n_hard][fixed].to(device)

        for i in range(self.n_iters):
            # --- Nesterov lookahead ---
            pos_look = (pos + mu * v).detach().requires_grad_(True)

            wl  = _wl_loss(pos_look, all_nodes, net_ids, n_nets, net_weights, gamma)
            den = _density_loss(pos_look, sizes, grid_res, bw, bh, cw, ch, device)

            # --- HPWL-plateau lambda ramp (transition point logic) ---
            wl_val = wl.item()
            wl_history.append(wl_val)
            if len(wl_history) > PLATEAU_WINDOW + 1:
                ref = wl_history[-(PLATEAU_WINDOW + 1)]
                improvement = (ref - wl_val) / (abs(ref) + 1e-10)
                if improvement < PLATEAU_THRESH and i > self.n_iters * 0.3:
                    density_weight = min(density_weight * LAMBDA_RAMP, LAMBDA_MAX)
       


            # --- Compute combined gradient ---
            total = wl + density_weight * den
            if pos_look.grad is not None:
                pos_look.grad.zero_()
            total.backward()

            with torch.no_grad():
                grad = pos_look.grad.clone()
                grad[fixed] = 0.0
                grad = grad / precond.unsqueeze(1)   # Jacobi preconditioning

                # --- Lipschitz constant estimation → adaptive lr (RePlAce §IV) ---
                # L = ||∇f_new - ∇f_old|| / ||x_new - x_old||, step = 1/L
                if prev_grad is not None and prev_pos_look is not None:
                    dg = (grad - prev_grad).norm().item()
                    dp = (pos_look.detach() - prev_pos_look).norm().item()
                    if dp > 1e-10 and dg > 1e-10:
                        L_local = dg / dp
                        lr = float(np.clip(1.0 / L_local, lr_min, lr_max))

                prev_grad     = grad.clone()
                prev_pos_look = pos_look.detach().clone()

                grad.clamp_(-(max(cw, ch) * 0.5), max(cw, ch) * 0.5)

                # --- Nesterov update ---
                v   = mu * v - lr * grad
                pos = pos + v
                pos[:, 0].clamp_(0.0, cw)
                pos[:, 1].clamp_(0.0, ch)
                if fixed.any():
                    pos[fixed] = fixed_pos

            if i % 50 == 0:
                print(f"  Iter {i:4d}: WL={wl_val:.4f}  Den={den.item():.6f}  "
                      f"λ={density_weight:.6f}  lr={lr:.6f}")

        # --- Legalize ---
        return self._finalize(pos.cpu(), benchmark)

    def _finalize(self, pos, benchmark):
        n = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
        movable = benchmark.get_movable_mask()[:n].numpy()
        sep_x, sep_y = _make_sep(sizes)

        full_pos = benchmark.macro_positions.clone()
        legalized = _legalize(
            pos[:n].numpy().astype(np.float64), movable,
            sizes, half_w, half_h, cw, ch, sep_x, sep_y,
        )
        full_pos[:n] = torch.tensor(legalized, dtype=torch.float32)
        return full_pos
