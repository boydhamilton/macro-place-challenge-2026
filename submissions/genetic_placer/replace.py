"""
RePlAce analytic global placement.

Implements the ePlace/RePlAce method:
  - WAWL: Weighted Average Wire Length — smooth, differentiable HPWL approximation
  - Bell-kernel bin density penalty to spread macros and prevent overlap
  - Nesterov (FISTA) accelerated gradient for fast convergence
  - Barzilai-Borwein step size with sufficient-decrease backtracking
  - Dynamic lambda schedule: grows based on HPWL increment per iteration
    (Algorithm 2 from RePlAce), not a fixed multiplicative rate
  - Gamma annealing: starts smooth, ends accurate
  - Relative overflow termination: threshold relative to initial overflow
  - Area-normalized gradient preconditioner for macro size differences

Based on:
  RePlAce: Advancing Solution Quality and Routability Validation in Global Placement
  Cheng et al., IEEE TCAD 2019.

  ePlace: Electrostatics-Based Placement Using Fast Fourier Transform and Nesterov's Method
  Lu et al., ACM TODAES 2015.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# WAWL — both gradient and scalar value
# ---------------------------------------------------------------------------

def _wawl_grad(pos, edges, weights, gamma):
    """
    Vectorized gradient of the Weighted Average Wire Length (WAWL) objective.

    WAWL is the smooth, differentiable HPWL approximation used by ePlace/RePlAce:
      W(e) = WA+(e) - WA-(e)
      WA+(e) = (Σ x_i exp(x_i/γ)) / (Σ exp(x_i/γ))   ≈ max_i x_i
      WA-(e) = (Σ x_i exp(-x_i/γ)) / (Σ exp(-x_i/γ))  ≈ min_i x_i

    For a 2-pin net between i at u and j at v, the gradient is:
      ∂W/∂u = exp_u/(γ·S+)·(γ + u − WA+)  −  exp_u_neg/(γ·S−)·(γ − u + WA−)
    Exponentials are shifted by max(u,v) / min(u,v) for numerical stability.
    """
    grad = np.zeros_like(pos)
    if len(edges) == 0:
        return grad

    i_idx = edges[:, 0]
    j_idx = edges[:, 1]

    for dim in range(2):
        xi = pos[i_idx, dim]  # [E]
        xj = pos[j_idx, dim]  # [E]

        # Positive direction (approximates max)
        x_max = np.maximum(xi, xj)
        ei = np.exp((xi - x_max) / gamma)
        ej = np.exp((xj - x_max) / gamma)
        s_pos = ei + ej
        wa_pos = (xi * ei + xj * ej) / s_pos

        gp_i = ei / (gamma * s_pos) * (gamma + xi - wa_pos)
        gp_j = ej / (gamma * s_pos) * (gamma + xj - wa_pos)

        # Negative direction (approximates min)
        x_min = np.minimum(xi, xj)
        ei_n = np.exp(-(xi - x_min) / gamma)
        ej_n = np.exp(-(xj - x_min) / gamma)
        s_neg = ei_n + ej_n
        wa_neg = (xi * ei_n + xj * ei_n) / s_neg

        gn_i = ei_n / (gamma * s_neg) * (gamma - xi + wa_neg)
        gn_j = ej_n / (gamma * s_neg) * (gamma - xj + wa_neg)

        dW_i = weights * (gp_i - gn_i)
        dW_j = weights * (gp_j - gn_j)

        np.add.at(grad[:, dim], i_idx, dW_i)
        np.add.at(grad[:, dim], j_idx, dW_j)

    return grad


def _wawl_value(pos, edges, weights, gamma):
    """
    Scalar WAWL objective value — used for lambda scheduling and
    sufficient-decrease check. Mirrors _wawl_grad but accumulates
    the scalar W(e) = WA+(e) - WA-(e) per edge.
    """
    if len(edges) == 0:
        return 0.0

    i_idx = edges[:, 0]
    j_idx = edges[:, 1]
    total = 0.0

    for dim in range(2):
        xi = pos[i_idx, dim]
        xj = pos[j_idx, dim]

        x_max = np.maximum(xi, xj)
        ei = np.exp((xi - x_max) / gamma)
        ej = np.exp((xj - x_max) / gamma)
        s_pos = ei + ej
        wa_pos = (xi * ei + xj * ej) / s_pos

        x_min = np.minimum(xi, xj)
        ei_n = np.exp(-(xi - x_min) / gamma)
        ej_n = np.exp(-(xj - x_min) / gamma)
        s_neg = ei_n + ej_n
        wa_neg = (xi * ei_n + xj * ej_n) / s_neg

        total += float((weights * (wa_pos - wa_neg)).sum())

    return total


# ---------------------------------------------------------------------------
# Bell-kernel bin density
# ---------------------------------------------------------------------------

def _density_overflow_grad(pos, sizes, half_w, half_h, movable, cw, ch, n_bins, target_density):
    """
    Bell-kernel bin density: total overflow and gradient.

    For each macro i and bin b:
      D_{b,i} = area_i · K_x(xi, bx, ax) · K_y(yi, by, ay)
    where K is the quadratic bell kernel  K(u, a) = max(0, 1 − (u/a)²)
    and  ax = half_w[i] + bin_w/2  (macro half-width + bin half-width).

    Density overflow:
      OF = Σ_b max(0, D_b − T_b) / total_movable_area

    Gradient:
      ∂OF/∂xi = Σ_{b: D_b > T_b}  ∂D_{b,i}/∂xi
    """
    bin_w = cw / n_bins
    bin_h = ch / n_bins
    target_area = target_density * bin_w * bin_h
    total_area = float((sizes[movable, 0] * sizes[movable, 1]).sum()) if len(movable) else 1.0

    bx_c = (np.arange(n_bins) + 0.5) * bin_w  # [B]
    by_c = (np.arange(n_bins) + 0.5) * bin_h  # [B]

    density = np.zeros((n_bins, n_bins))

    # Forward: accumulate density from each macro
    for idx in movable:
        ax = half_w[idx] + bin_w / 2.0
        ay = half_h[idx] + bin_h / 2.0
        xi, yi = pos[idx, 0], pos[idx, 1]
        area_i = sizes[idx, 0] * sizes[idx, 1]

        bxi_lo = max(0, int((xi - ax) / bin_w))
        bxi_hi = min(n_bins - 1, int((xi + ax) / bin_w))
        byi_lo = max(0, int((yi - ay) / bin_h))
        byi_hi = min(n_bins - 1, int((yi + ay) / bin_h))

        bx = bx_c[bxi_lo : bxi_hi + 1]
        by = by_c[byi_lo : byi_hi + 1]
        ux = xi - bx
        uy = yi - by
        kx = np.maximum(0.0, 1.0 - (ux / ax) ** 2)
        ky = np.maximum(0.0, 1.0 - (uy / ay) ** 2)
        density[byi_lo : byi_hi + 1, bxi_lo : bxi_hi + 1] += np.outer(ky, kx) * area_i

    overflow_map = np.maximum(0.0, density - target_area)  # [B, B]
    overflow = float(overflow_map.sum()) / max(total_area, 1e-9)

    # Backward: accumulate gradient from overflowing bins
    grad = np.zeros_like(pos)
    for idx in movable:
        ax = half_w[idx] + bin_w / 2.0
        ay = half_h[idx] + bin_h / 2.0
        xi, yi = pos[idx, 0], pos[idx, 1]
        area_i = sizes[idx, 0] * sizes[idx, 1]

        bxi_lo = max(0, int((xi - ax) / bin_w))
        bxi_hi = min(n_bins - 1, int((xi + ax) / bin_w))
        byi_lo = max(0, int((yi - ay) / bin_h))
        byi_hi = min(n_bins - 1, int((yi + ay) / bin_h))

        bx = bx_c[bxi_lo : bxi_hi + 1]
        by = by_c[byi_lo : byi_hi + 1]
        ux = xi - bx
        uy = yi - by
        kx  = np.maximum(0.0, 1.0 - (ux / ax) ** 2)
        ky  = np.maximum(0.0, 1.0 - (uy / ay) ** 2)
        dkx = np.where(np.abs(ux) < ax, -2.0 * ux / (ax * ax), 0.0)
        dky = np.where(np.abs(uy) < ay, -2.0 * uy / (ay * ay), 0.0)

        ov = overflow_map[byi_lo : byi_hi + 1, bxi_lo : bxi_hi + 1]  # [By, Bx]
        grad[idx, 0] += float((ov * np.outer(ky,  dkx) * area_i).sum())
        grad[idx, 1] += float((ov * np.outer(dky,  kx) * area_i).sum())

    return grad, overflow


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _run_replace(
    movable_idx, movable_mask, sizes, half_w, half_h,
    cw, ch, init_pos, areas, edges, weights, n,
    n_iters=200, gamma_frac=0.01, n_bins=16, target_density=1.0,
):
    """
    RePlAce analytic global placement.

    Minimizes  f(x) = WAWL(x) + λ · density_penalty(x)
    using Nesterov's accelerated gradient (FISTA):

      a_0 = 1,  v_0 = x_0
      for k = 0, 1, ...:
        x_{k+1} = clip( v_k − α_k · ∇f(v_k) )
        a_{k+1} = (1 + √(1 + 4·a_k²)) / 2
        coeff   = (a_k − 1) / a_{k+1}
        v_{k+1} = clip( x_{k+1} + coeff·(x_{k+1} − x_k) )

    Step size α is updated via the Barzilai-Borwein secant rule with a
    sufficient-decrease backtracking guard to prevent overestimation.

    Lambda schedule follows RePlAce Algorithm 2: cof is dynamically adjusted
    each iteration to maintain a roughly constant HPWL increment, rather than
    growing at a fixed rate. This discerns between early (fast-moving) and
    late (settling) stages of placement.

    Gamma is annealed from smooth (large) to accurate (small) over iterations,
    improving the WAWL approximation quality as placement converges.

    Overflow termination is relative to initial overflow (τ_init / 2.5),
    matching the paper's trial placement termination condition.

    Gradient is area-normalized (preconditioned) to handle macro size
    differences without manual per-macro tuning.

    Returns positions of shape (n, 2) ready for _legalize.
    """
    pos = init_pos.copy()
    movable = np.array(movable_idx)

    if len(movable) == 0:
        return pos

    # Clip initial positions into canvas
    pos[movable, 0] = np.clip(pos[movable, 0], half_w[movable], cw - half_w[movable])
    pos[movable, 1] = np.clip(pos[movable, 1], half_h[movable], ch - half_h[movable])

    canvas = max(cw, ch)

    # --- Gamma schedule: anneal from smooth to accurate ---
    # Large gamma early: smooth landscape, easier to optimize globally.
    # Small gamma late: accurate HPWL approximation as macros settle.
    gamma_start = gamma_frac * (cw + ch) / 2.0
    gamma_end   = gamma_start * 0.1
    gamma       = gamma_start

    # --- Area-based preconditioner: normalize gradient by macro area ---
    # Prevents large macros from dominating gradient magnitude and
    # drowning out the signal for smaller macros (RePlAce §II preconditioner).
    area_scale = 1.0 / np.maximum(areas[:n], 1e-9)  # [n]

    # --- Initial lambda: balance wirelength and density gradient magnitudes ---
    g_wl  = _wawl_grad(pos, edges, weights, gamma)
    g_den, overflow_init = _density_overflow_grad(
        pos, sizes, half_w, half_h, movable, cw, ch, n_bins, target_density
    )

    # Apply preconditioner to initial gradients for fair norm comparison
    g_wl_scaled  = g_wl.copy()
    g_den_scaled = g_den.copy()
    g_wl_scaled[movable]  *= area_scale[movable, np.newaxis]
    g_den_scaled[movable] *= area_scale[movable, np.newaxis]

    norm_wl  = float(np.linalg.norm(g_wl_scaled[movable]))  if len(edges) > 0 else 0.0
    norm_den = float(np.linalg.norm(g_den_scaled[movable]))
    lam = (norm_wl / norm_den) if norm_den > 1e-12 else 1.0

    # --- Relative overflow termination (RePlAce §IV-B trial placement) ---
    # Paper terminates when overflow <= tau_init / 2.5.
    # Guards against hanging on dense benchmarks (absolute 0.1 can be unreachable)
    # and premature stop on sparse ones.
    overflow_threshold = max(overflow_init / 2.5, 0.05)

    # --- Lambda dynamic schedule parameters (RePlAce Algorithm 2) ---
    # cof varies in [cof_min, cof_max] based on HPWL increment per iteration.
    # Larger HPWL increment → smaller cof (slow lambda growth to let WL recover).
    # Smaller HPWL increment → larger cof (fast lambda growth to enforce spread).
    cof_min = 0.95
    cof_max = 1.05
    hpwl_ref = _wawl_value(pos, edges, weights, gamma) * 0.001  # target ~0.1% change
    hpwl_ref = max(hpwl_ref, 1e-6)
    prev_hpwl = _wawl_value(pos, edges, weights, gamma)

    # --- Nesterov / FISTA state ---
    x_k   = pos.copy()
    v_k   = pos.copy()
    a_k   = 1.0
    alpha = canvas * 0.005  # initial step; BB rule self-tunes from iter 1 onward

    prev_grad = (g_wl + lam * g_den).copy()
    prev_v    = v_k.copy()

    for it in range(n_iters):
        # --- Gamma annealing ---
        progress = it / max(n_iters - 1, 1)
        gamma = gamma_start * (gamma_end / gamma_start) ** progress

        # --- Compute gradients at current Nesterov point v_k ---
        g_wl  = _wawl_grad(v_k, edges, weights, gamma)
        g_den, overflow = _density_overflow_grad(
            v_k, sizes, half_w, half_h, movable, cw, ch, n_bins, target_density
        )
        grad = g_wl + lam * g_den

        # --- Area preconditioner: scale gradient by inverse macro area ---
        grad[movable] *= area_scale[movable, np.newaxis]

        # --- Barzilai-Borwein step size (skip iter 0 — no previous state) ---
        if it > 0:
            dv = v_k[movable] - prev_v[movable]
            dg = grad[movable] - prev_grad[movable]
            dv_sq = float((dv * dv).sum())
            dv_dg = abs(float((dv * dg).sum()))
            if dv_sq > 1e-12 and dv_dg > 1e-12:
                alpha = float(np.clip(
                    dv_sq / dv_dg,
                    canvas * 1e-5,
                    canvas * 0.05,
                ))

        prev_grad = grad.copy()
        prev_v    = v_k.copy()

        # --- Gradient step: x_{k+1} = clip(v_k − α · ∇f) ---
        x_k1 = v_k.copy()
        x_k1[movable, 0] = np.clip(
            v_k[movable, 0] - alpha * grad[movable, 0],
            half_w[movable], cw - half_w[movable],
        )
        x_k1[movable, 1] = np.clip(
            v_k[movable, 1] - alpha * grad[movable, 1],
            half_h[movable], ch - half_h[movable],
        )

        # --- Sufficient-decrease backtracking guard ---
        # If the objective increased significantly, the BB step overshot.
        # Halve alpha and recompute. One backtrack step is enough for stability
        # without the cost of a full line search.
        curr_hpwl = _wawl_value(x_k1, edges, weights, gamma)
        _, overflow_new = _density_overflow_grad(
            x_k1, sizes, half_w, half_h, movable, cw, ch, n_bins, target_density
        )
        f_new = curr_hpwl + lam * overflow_new
        f_old = prev_hpwl + lam * overflow

        if f_new > f_old * 1.5 and it > 0:
            alpha *= 0.5
            x_k1[movable, 0] = np.clip(
                v_k[movable, 0] - alpha * grad[movable, 0],
                half_w[movable], cw - half_w[movable],
            )
            x_k1[movable, 1] = np.clip(
                v_k[movable, 1] - alpha * grad[movable, 1],
                half_h[movable], ch - half_h[movable],
            )
            curr_hpwl = _wawl_value(x_k1, edges, weights, gamma)

        # --- Dynamic lambda schedule (RePlAce Algorithm 2) ---
        # cof is driven by the ratio of actual HPWL increment to reference,
        # not a fixed 1.02 rate. This allocates more lambda pressure when
        # HPWL is stable, less when it's rising fast.
        hpwl_delta = abs(curr_hpwl - prev_hpwl)
        p = hpwl_delta / hpwl_ref
        if p < 0:
            cof = cof_max
        else:
            cof = max(cof_min, cof_max ** (1.0 - p))
        lam *= cof
        prev_hpwl = curr_hpwl

        # --- Nesterov momentum: a_{k+1} = (1 + √(1 + 4·a_k²)) / 2 ---
        a_k1  = (1.0 + math.sqrt(1.0 + 4.0 * a_k * a_k)) / 2.0
        coeff = (a_k - 1.0) / a_k1

        # --- Momentum update: v_{k+1} = clip(x_{k+1} + coeff·(x_{k+1} − x_k)) ---
        v_k1 = x_k1.copy()
        v_k1[movable] = x_k1[movable] + coeff * (x_k1[movable] - x_k[movable])
        v_k1[movable, 0] = np.clip(v_k1[movable, 0], half_w[movable], cw - half_w[movable])
        v_k1[movable, 1] = np.clip(v_k1[movable, 1], half_h[movable], ch - half_h[movable])

        x_k = x_k1
        v_k = v_k1
        a_k = a_k1

        # --- Relative overflow termination (paper §IV-B) ---
        if overflow < overflow_threshold:
            break

    return x_k