#!/usr/bin/env python3
"""
Sign convention (VAH01 logs charge current as POSITIVE):
    V = OCV(SoC) + I*R0(SoC) + V1,         tau * dV1/dt = -V1 + Rp(SoC)*I
    V1[k+1] = a*V1[k] + Rp*(1 - a)*I[k],   a = exp(-dt/tau)        (exact zero-order hold, as in Step 1)
The spec's "OCV - I*R0 - ..." is the discharge-positive form of the same model.

Pipeline (numbers match the pseudocode):
  0  inventory: segment the file, find the BOL RPT block, its pulses and the early 1C mission charges
  1  coulomb counting: Q_BOL and SoC from our own integral of measured I (not the Q columns)
  2  pulse fits  -> R0, Rp, tau (+ an OCV-drift term) at each pulse SoC
  3  two-rate long-time resistance  R_cc(SoC) = (V_1C - V_C/2) / (I_1C - I_C/2)
  4  OCV(SoC) on the charge branch = V_C/2 - I_C/2 * R_total(SoC), monotone fit
  5  optional joint refinement (off by default, --joint-refine)
  6  hand-off bundle  <cell>_ecm_bol.json
  validation tiers A-E on measured V/I with "oracle" SoC from measured I   [DEV-ONLY]

Deployment boundary: everything here uses measured V/I and is [DEV-ONLY] / one-time calibration.
What deployment may use is only the frozen hand-off bundle (OCV, R_total_BOL, phi, tau, Q_BOL).

Tags in comments: [DATA] read off the VAH01 files (see CLAUDE.md), [ASSUMED] my choice where nothing
fixes it, [DEV-ONLY] uses measured V/I in a way that never belongs on the deployment path.

Usage:
    python voltage_with_ecm.py --csv VAH01.csv --impedance VAH01_impedance.csv --out step2_results
    python voltage_with_ecm.py --csv VAH01.csv --inventory-only      # look before fitting
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Settings (module-level constants; some are overridden from the CLI in main())
# ============================================================================

# --- Data conventions ---
CHARGE_POSITIVE = True          # [DATA] VAH01 logs charge current as positive
CC_CURRENT_A = 3.0              # [DATA] mission CC charge current (1C), Ns 0
RPT_CURRENT_A = 1.5             # [DATA] RPT CC charge current (C/2), logged every 10 s
CURRENT_REL_TOL = 0.10          # [ASSUMED] |I - I_nom| / I_nom accepted as "at that CC level"
REST_CURRENT_A = 0.02           # [ASSUMED] |I| below this counts as rest
MIN_CC_SAMPLES = 20             # [ASSUMED] same as Step 1
RPT_MIN_DURATION_S = 1800.0     # [ASSUMED] a C/2 CC run longer than this is an RPT charge
EMPTY_V = 3.0                   # [ASSUMED] a discharge whose voltage drops below this reached "empty"
Q_SOURCE = "auto"               # [ASSUMED] "auto" | "discharge" | "charge": which RPT leg defines Q

# --- Pulse detection and fit (pseudocode step 2) ---
PULSE_MIN_S = 1.0               # [ASSUMED]
PULSE_MAX_S = 120.0             # [ASSUMED] the impedance file implies >= 30 s pulses
PULSE_MIN_DI_A = 0.3            # [ASSUMED] current change that counts as a step (VAH01 RPT: ~0.53 A)
PRE_MIN_S = 300.0               # [ASSUMED] the run before a step must be long enough for V1 to be steady
PRE_FIT_S = 300.0               # [ASSUMED] seconds of the run before the step included in the fit
PRE_REST_AVG_S = 10.0           # [ASSUMED] V_pre = mean V over the last this-many seconds before the step
POST_FIT_S = 300.0              # [ASSUMED] seconds of the run after the step included in the fit
IMPEDANCE_LABEL = "dod"         # [DATA] VAH01 impedance columns "20%"/"60%" are depth of discharge
IMPEDANCE_SOC_TOL = 0.15        # [ASSUMED]
TAU_STARTS_S = (5.0, 20.0, 60.0)    # [ASSUMED] multi-start for tau
TAU_BOUNDS_S = (0.5, 600.0)     # [ASSUMED]
DRIFT_BOUNDS_V = (0.0, 3.0)     # [ASSUMED] dOCV/dSoC in V per unit SoC (3 V = 30 mV per %)
PULSE_MAX_RMSE_V = 0.005        # [ASSUMED] pulse fits worse than this are not used
PULSE_SOC_CLUSTER = 0.05        # [ASSUMED] pulses closer than this in SoC are averaged
TAU_FALLBACK_S = 30.0           # [ASSUMED] used only if no pulse fit is usable
TAU_PER_SOC = False             # [ASSUMED] False: one median tau; True: tau(SoC) between pulse SoCs

# --- Two-rate resistance and OCV (pseudocode steps 3-4) ---
SOC_GRID_N = 201                # [ASSUMED] grid on [0, 1] used for every curve in the hand-off
TRIM_TAU_MULT = 3.0             # [ASSUMED] drop the first ~3 tau of each CC (V1 still building)
TRIM_MIN_S = 60.0               # [ASSUMED]
RCC_TRIM_S = 600.0              # [ASSUMED] longer trim for the two-rate difference: slow polarization
                                # (diffusion) needs minutes to build, and mission charges start at 25-45% SoC
N_FIT_EVENTS = 10               # [ASSUMED] early 1C mission charges used for R_cc
N_VAL_EVENTS = 10               # [ASSUMED] the next ones, held out for validation tier B/E
MIN_EVENTS_PER_BIN = 3          # [ASSUMED]
RCC_SMOOTH_BINS = 5             # [ASSUMED] moving average over SoC-grid bins
RCC_MATCH_TOL = 0.15            # [ASSUMED] |R_cc / R_pulse - 1| below this -> pulse tau is kept
OCV_KNOTS = 12                  # [ASSUMED] fewer knots = stronger implicit smoothing
USE_POST_RPT_1C = False         # [ASSUMED] the first 1C after an RPT starts from empty after a 21 C rest

# --- Optional joint refinement (pseudocode step 5) ---
RUN_JOINT_REFINE = False        # [ASSUMED] only worth running if residuals are SoC-structured
SMOOTHNESS_WEIGHT = 1e-2        # [ASSUMED] "lambda": weight on second differences of the curves
PRIOR_WEIGHT = 1e-2             # [ASSUMED] "mu": weight keeping theta near the step 2-4 estimates
R_KNOTS = 6                     # [ASSUMED]

# --- Validation ---
BOOTSTRAP_N = 200               # [ASSUMED]
SOC0_PERTURB = (0.01, 0.02, 0.05)   # [ASSUMED] tier E
RTOT_PERTURB = 0.10             # [ASSUMED] tier E
PHI_PERTURB = 0.10              # [ASSUMED] tier E
AGING_STRIDE = 10               # [ASSUMED] tier D: every Nth mission charge
V_CV_DEFAULT = 4.2              # [DATA] CV voltage if a charge has no CV step to read it from
CV_EXTEND_MAX_S = 1800.0        # [ASSUMED] how far past the measured CC to look for the CV crossing
RANDOM_SEED = 0
MAKE_PLOTS = True


# ============================================================================
# 1. Small helpers
# ============================================================================

def funcruns(x):
    """Half-open index ranges [start, stop) of consecutive equal values (same idea as Step 1)."""
    x = np.asarray(x)
    if x.size == 0:
        return []
    b = np.flatnonzero(x[1:] != x[:-1]) + 1
    return list(zip(np.r_[0, b].tolist(), np.r_[b, x.size].tolist()))


def seg_duration(t, start, stop):
    """Duration of samples [start, stop), measured to the next sample when there is one."""
    end = stop if stop < len(t) else stop - 1
    return float(t[end] - t[start])


def rmse(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def interp_on_grid(x, y, grid):
    """Interpolate y(x) onto grid, NaN outside the data range. Duplicate x are averaged."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return np.full(len(grid), np.nan)
    xu, inv = np.unique(x, return_inverse=True)
    yu = np.bincount(inv, weights=y) / np.bincount(inv)
    if xu.size < 2:
        return np.full(len(grid), np.nan)
    out = np.interp(grid, xu, yu)
    out[(grid < xu[0]) | (grid > xu[-1])] = np.nan
    return out


def interp_extrap(x, xp, fp):
    """Linear interpolation with linear extrapolation from the end segments."""
    x = np.atleast_1d(np.asarray(x, float))
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    if lo.any():
        y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    if hi.any():
        y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


def nan_moving_average(y, w):
    """Centred moving average that ignores NaN and keeps NaN where y is NaN."""
    y = np.asarray(y, float)
    if w <= 1:
        return y.copy()
    ok = np.isfinite(y)
    k = np.ones(int(w))
    num = np.convolve(np.where(ok, y, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    out = np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
    out[~ok] = np.nan
    return out


def nanmedian_quiet(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def nanpercentile_quiet(a, q, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(a, q, axis=axis)


def pava_increasing(y):
    """Pool-adjacent-violators: the closest non-decreasing sequence in least squares."""
    vals, wts, lens = [], [], []
    for v in np.asarray(y, float):
        vals.append(v)
        wts.append(1.0)
        lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            n2 = lens[-2] + lens[-1]
            vals[-2:], wts[-2:], lens[-2:] = [v2], [w], [n2]
    return np.repeat(vals, lens)


def hat_basis(x, knots):
    """Piecewise-linear (hat-function) basis on knots, for a linear least-squares curve fit."""
    x = np.clip(np.asarray(x, float), knots[0], knots[-1])
    K = len(knots)
    j = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, K - 2)
    w = np.clip((x - knots[j]) / (knots[j + 1] - knots[j]), 0.0, 1.0)
    H = np.zeros((len(x), K))
    rows = np.arange(len(x))
    H[rows, j] = 1.0 - w
    H[rows, j + 1] += w
    return H


def rc_response(t, I, Rp, tau, v1_0=0.0):
    """V1 of one R||C pair under zero-order-hold current (Rp, tau scalars or per-sample arrays)."""
    n = len(t)
    v1 = np.empty(n)
    v1[0] = v1_0
    if n == 1:
        return v1
    Rp = np.broadcast_to(np.asarray(Rp, float), (n,))
    tau = np.broadcast_to(np.asarray(tau, float), (n,))
    a = np.exp(-np.diff(t) / tau[:-1])
    b = (Rp[:-1] * (1.0 - a) * I[:-1]).tolist()
    a = a.tolist()
    v = float(v1_0)
    for k in range(n - 1):              # plain floats: much faster than numpy scalar indexing
        v = a[k] * v + b[k]
        v1[k + 1] = v
    return v1


# ============================================================================
# 2. Loading and segmentation (pseudocode step 0)
# ============================================================================

def load_cell(path):
    df = pd.read_csv(path)
    need = ["time_s", "Ecell_V", "I_mA", "Temperature__C", "Ns"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path}: missing columns {miss}")
    df = df.sort_values("time_s", kind="stable").reset_index(drop=True)   # [DATA] duplicate stamps exist
    t = df["time_s"].to_numpy(float)
    I = df["I_mA"].to_numpy(float) / 1000.0
    if not CHARGE_POSITIVE:
        I = -I
    # Own coulomb count (trapezoid), in Ah. Not the logged Q columns, which may reset.
    cum_ah = np.r_[0.0, np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t))] / 3600.0
    return dict(name=os.path.splitext(os.path.basename(path))[0], t=t, I=I,
                V=df["Ecell_V"].to_numpy(float), T=df["Temperature__C"].to_numpy(float),
                ns=df["Ns"].to_numpy(), cum_ah=cum_ah, n=len(t))


def find_mission_events(cell):
    """1C CC charges (Ns 0 at ~CC_CURRENT_A), chronological, with the CV run that follows."""
    t, I, V, T, ns = cell["t"], cell["I"], cell["V"], cell["T"], cell["ns"]
    runs = funcruns(ns)
    events = []
    for j, (s, e) in enumerate(runs):
        if ns[s] != 0:
            continue
        inband = np.abs(I[s:e] - CC_CURRENT_A) <= CURRENT_REL_TOL * CC_CURRENT_A
        if inband.sum() < MIN_CC_SAMPLES:
            continue
        idx = np.flatnonzero(inband)
        s2, e2 = s + int(idx[0]), s + int(idx[-1]) + 1
        cv_stop, v_cv = None, np.nan
        if j + 1 < len(runs) and ns[runs[j + 1][0]] == 1:              # [DATA] Ns 1 = CV
            cs, ce = runs[j + 1]
            cv_stop, v_cv = int(ce), float(np.median(V[cs:ce]))
        events.append(dict(k=len(events), start=s2, stop=e2, cv_stop=cv_stop, v_cv=v_cv,
                           t0=float(t[s2]), I=float(I[s2:e2].mean()), T=float(T[s2:e2].mean()),
                           after_rpt=False))
    return events


def find_rpt_blocks(cell, events):
    """Each long C/2 CC charge marks an RPT. The block runs from the end of the previous mission
    charge to the start of the next one, so it contains the RPT's rests, pulses and discharge."""
    t, I, V, T, cum, n = cell["t"], cell["I"], cell["V"], cell["T"], cell["cum_ah"], cell["n"]
    inband = (I > REST_CURRENT_A) & (np.abs(I - RPT_CURRENT_A) <= CURRENT_REL_TOL * RPT_CURRENT_A)
    ev_starts = np.array([e["start"] for e in events], dtype=int)
    ev_ends = np.array([e["cv_stop"] if e["cv_stop"] is not None else e["stop"] for e in events], dtype=int)
    blocks = []
    for s, e in funcruns(inband):
        if not inband[s] or seg_duration(t, s, e) < RPT_MIN_DURATION_S:
            continue
        k = e
        while k < n and I[k] > REST_CURRENT_A:      # follow the charge through its CV step
            k += 1
        full = k - 1
        prev = ev_ends[ev_ends <= s]
        b0 = int(prev.max()) if prev.size else 0
        nxt = ev_starts[ev_starts > full]
        b1 = int(nxt.min()) if nxt.size else n
        q_chg = float(cum[full] - cum[b0:s + 1].min())
        has_dis = b1 > full + 1
        q_dis = float(cum[full] - cum[full:b1].min()) if has_dis else np.nan
        dis_empty = bool(has_dis and V[full:b1].min() < EMPTY_V)
        chg_empty = bool(V[b0:s + 1].min() < EMPTY_V)
        if (Q_SOURCE == "discharge" or (Q_SOURCE == "auto" and dis_empty)) and np.isfinite(q_dis):
            q, src = q_dis, "discharge"
        elif Q_SOURCE == "charge" or (Q_SOURCE == "auto" and chg_empty):
            q, src = q_chg, "charge"
        else:
            q, src = float(np.nanmax([q_dis, q_chg])), "max (neither leg reached EMPTY_V)"
        blocks.append(dict(b0=b0, cc_start=int(s), cc_stop=int(e), full=int(full), b1=b1,
                           t_full=float(t[full]), I=float(I[s:e].mean()), T=float(T[s:e].mean()),
                           q=q, q_chg=q_chg, q_dis=q_dis, q_src=src, dis_empty=dis_empty,
                           chg_empty=chg_empty, cycle=int(np.sum(ev_starts < s))))
        after = [ev for ev in events if ev["start"] > full]
        if after:
            after[0]["after_rpt"] = True            # [DATA] starts from empty after a long rest
    return blocks


def full_indices(events, blocks):
    """Sample indices where the cell is taken to be full (end of a CV step). [ASSUMED] SoC = 1 there."""
    f = [ev["cv_stop"] - 1 for ev in events if ev["cv_stop"] is not None] + [b["full"] for b in blocks]
    return np.array(sorted(f), dtype=int)


def block_soc(cell, blk, idx, q=None):
    """SoC inside an RPT block, anchored at SoC = 1 at the end of its CV charge."""
    q = blk["q"] if q is None else q
    return 1.0 - (cell["cum_ah"][blk["full"]] - cell["cum_ah"][idx]) / q


def soc0_forward(cell, fulls, start, q):
    """Known-duty-cycle SoC0: 1 - (Ah taken out since the last full charge) / Q."""
    j = int(np.searchsorted(fulls, start)) - 1
    if j < 0:
        return np.nan
    return float(1.0 - (cell["cum_ah"][fulls[j]] - cell["cum_ah"][start]) / q)


def soc0_backward(cell, ev, q):
    """Consistency check: SoC0 from the Ah put back in up to the end of this charge's own CV."""
    if ev["cv_stop"] is None:
        return np.nan
    return float(1.0 - (cell["cum_ah"][ev["cv_stop"] - 1] - cell["cum_ah"][ev["start"]]) / q)


def event_soc(cell, ev, soc0, q):
    idx = np.arange(ev["start"], ev["stop"])
    return soc0 + (cell["cum_ah"][idx] - cell["cum_ah"][ev["start"]]) / q


def print_inventory(cell, events, blocks, max_runs=80):
    t, I, V, T, ns = cell["t"], cell["I"], cell["V"], cell["T"], cell["ns"]
    print(f"\n=== Inventory: {cell['name']} ({cell['n']} rows, {(t[-1] - t[0]) / 3600:.1f} h) ===")
    print(f"1C mission CC events: {len(events)} ({sum(e['after_rpt'] for e in events)} first-after-RPT)")
    print(f"RPT C/2 charges found: {len(blocks)}")
    for i, b in enumerate(blocks):
        print(f"  RPT {i}: t_full={b['t_full'] / 3600:8.1f} h  cycle~{b['cycle']:4d}  Q={b['q']:.4f} Ah "
              f"({b['q_src']}; chg {b['q_chg']:.4f}, dis {b['q_dis']:.4f})  I={b['I']:.2f} A  T={b['T']:.1f} C")
    if not blocks:
        return
    b = blocks[0]
    print(f"\nBOL block, samples {b['b0']}..{b['b1']}: runs of equal Ns")
    print(f"{'Ns':>4} {'t_start_h':>9} {'dur_s':>8} {'n':>6} {'dt_med':>7} {'I_mean':>7} "
          f"{'V_min':>6} {'V_max':>6} {'T_mean':>6}")
    runs = [(s + b["b0"], e + b["b0"]) for s, e in funcruns(ns[b["b0"]:b["b1"]])]
    for s, e in runs[:max_runs]:
        dtm = float(np.median(np.diff(t[s:e]))) if e - s > 1 else np.nan
        print(f"{ns[s]!s:>4} {t[s] / 3600:9.3f} {seg_duration(t, s, e):8.0f} {e - s:6d} {dtm:7.1f} "
              f"{I[s:e].mean():7.3f} {V[s:e].min():6.3f} {V[s:e].max():6.3f} {T[s:e].mean():6.1f}")
    if len(runs) > max_runs:
        print(f"  ... {len(runs) - max_runs} more runs")
    pulses = find_pulses(cell, b["full"], b["b1"])
    print(f"\nSteps/pulses detected after the BOL RPT charge: {len(pulses)}")
    for p in pulses:
        soc = block_soc(cell, b, np.array([p["start"]]))[0]
        print(f"  {p['kind']:<4} t={t[p['start']] / 3600:8.3f} h  SoC~{soc:.3f}  I {p['I_prev']:+.3f} -> {p['I']:+.3f} A  "
              f"dur={p['dur']:.0f} s  dt_med={p['dt_med']:.1f} s  V_pre={p['v_pre']:.4f} V  "
              f"after-window samples={p['win1'] - p['stop']}")


# ============================================================================
# 3. ECM model and its hand-off form
# ============================================================================

class EcmModel:
    """Frozen BOL ECM on a SoC grid. Charge-positive: V = OCV + I*R0 + V1."""

    def __init__(self, grid, ocv, r_total, phi, tau, soc_valid, q_ah, t_ref):
        self.grid = np.asarray(grid, float)
        self.ocv = np.asarray(ocv, float)
        self.r_total = np.asarray(r_total, float)
        self.phi = np.asarray(phi, float)
        self.tau = np.broadcast_to(np.asarray(tau, float), self.grid.shape).copy()
        self.soc_valid = (float(soc_valid[0]), float(soc_valid[1]))
        self.q_ah = float(q_ah)
        self.t_ref = float(t_ref)

    def ocv_at(self, s):
        return interp_extrap(s, self.grid, self.ocv)

    def simulate(self, t, I, soc, v1_0=0.0, r_scale=1.0, phi_shift=0.0, zero_rc=False):
        s = np.atleast_1d(np.asarray(soc, float))
        rt = np.interp(s, self.grid, self.r_total) * r_scale
        if zero_rc:                     # 0-RC baseline: all resistance acts instantly
            return self.ocv_at(s) + I * rt
        phi = np.clip(np.interp(s, self.grid, self.phi) + phi_shift, 0.0, 1.0)
        tau = np.interp(s, self.grid, self.tau)
        return self.ocv_at(s) + I * phi * rt + rc_response(t, I, (1.0 - phi) * rt, tau, v1_0)

    def to_dict(self):
        r0 = self.phi * self.r_total
        return dict(soc_grid=self.grid.tolist(), ocv_V=self.ocv.tolist(), r_total_ohm=self.r_total.tolist(),
                    phi=self.phi.tolist(), r0_ohm=r0.tolist(), rp_ohm=(self.r_total - r0).tolist(),
                    tau_s=self.tau.tolist(), soc_valid=list(self.soc_valid), q_bol_Ah=self.q_ah,
                    t_ref_C=self.t_ref)


# ============================================================================
# 4. Pulse fits (pseudocode step 2)
# ============================================================================

PARAM_NAMES = ("R0", "Rp", "ltau", "drift", "c0")


def find_pulses(cell, i0, i1):
    """Short current steps inside samples [i0, i1): short Ns runs whose current differs from the long
    run before them. Two kinds:
      'rest' : the run before is a rest -> V0 (mean of its last seconds) is a measured OCV point
      'step' : the run before carries current (VAH01 RPT: C/5 discharge -> 30 s at ~C/40 -> C/5)
               -> the OCV level c0 is fitted, and V1 starts at its steady state Rp*I_prev."""
    t, I, V, T, ns = cell["t"], cell["I"], cell["V"], cell["T"], cell["ns"]
    runs = [(s + i0, e + i0) for s, e in funcruns(ns[i0:i1])]
    pulses = []
    for j in range(1, len(runs)):
        s, e = runs[j]
        ps, pe = runs[j - 1]
        dur = seg_duration(t, s, e)
        if not (PULSE_MIN_S <= dur <= PULSE_MAX_S) or e - s < 3:
            continue
        i_p, i_prev = float(np.median(I[s:e])), float(np.median(I[ps:pe]))
        if abs(i_p - i_prev) < PULSE_MIN_DI_A or t[s] - t[ps] < PRE_MIN_S:
            continue
        w0 = ps + int(np.searchsorted(t[ps:pe], t[s] - PRE_FIT_S, side="left"))
        w1 = e
        if j + 1 < len(runs):
            rs, re_ = runs[j + 1]
            w1 = rs + int(np.searchsorted(t[rs:re_], t[rs] + POST_FIT_S, side="right"))
        pre = ps + np.flatnonzero(t[ps:pe] >= t[pe - 1] - PRE_REST_AVG_S)
        pulses.append(dict(kind="rest" if abs(i_prev) < REST_CURRENT_A else "step", start=int(s), stop=int(e),
                           win0=int(w0), win1=int(w1), I=i_p, I_prev=i_prev, dur=dur,
                           v_pre=float(V[pre].mean()), T=float(T[s:e].mean()),
                           dt_med=float(np.median(np.diff(t[s:e])))))
    return pulses


def pulse_window(cell, p, soc_fn):
    """Fit window. The cycler logs the step instant with a mid-transition current and voltage: that sample
    gets the new run's current (zero-order hold from that instant) and its voltage is left out of the fit."""
    idx = np.arange(p["win0"], p["win1"])
    t = cell["t"][idx] - cell["t"][p["start"]]
    I, V, ns = cell["I"][idx].copy(), cell["V"][idx], cell["ns"][idx]
    w = np.ones(len(idx), bool)
    for k in np.flatnonzero(np.r_[False, ns[1:] != ns[:-1]]):
        j = k + 1
        while j < len(idx) and ns[j] == ns[k] and j < k + 6:
            j += 1
        if j > k + 1:
            I[k] = np.median(I[k + 1:j])
        w[k] = False
    soc = soc_fn(idx)
    return t, I, V, soc - soc_fn(np.array([p["start"]]))[0], w


def pulse_predict(x, t, I, dsoc):
    r0, rp, ltau, drift, c0 = x
    # V1 starts at steady state for the current before the window (0 after a rest).
    # drift*dsoc: OCV moves during the window, else Rp/tau are biased.
    return c0 + I * r0 + rc_response(t, I, rp, np.exp(ltau), v1_0=rp * I[0]) + drift * dsoc


def fit_pulse(t, I, V, dsoc, w, c0_init, fixed=None):
    """Least squares on pre-window, step and recovery jointly. Multi-start in tau."""
    fixed = dict(fixed or {})
    free = np.array([i for i, nm in enumerate(PARAM_NAMES) if nm not in fixed], dtype=int)
    lo = np.array([1e-5, 0.0, np.log(TAU_BOUNDS_S[0]), DRIFT_BOUNDS_V[0], 2.0])
    hi = np.array([1.0, 1.0, np.log(TAU_BOUNDS_S[1]), DRIFT_BOUNDS_V[1], 4.6])
    xs = np.array([0.01, 0.01, 1.0, 0.5, 0.01])

    def full(z):
        x = np.array([fixed.get(nm, 0.0) for nm in PARAM_NAMES], float)
        x[free] = z
        return x

    def resid(z):
        return (pulse_predict(full(z), t, I, dsoc) - V)[w]

    best = None
    for tau0 in TAU_STARTS_S:
        z0 = np.array([0.02, 0.02, np.log(tau0), 0.5, c0_init])[free]
        z0 = np.clip(z0, lo[free] + 1e-9, hi[free] - 1e-9)
        r = least_squares(resid, z0, bounds=(lo[free], hi[free]), x_scale=xs[free])
        if best is None or r.cost < best.cost:
            best = r
    x = full(best.x)
    std = np.full(5, np.nan)
    dof = int(w.sum()) - len(free)
    if dof > 0:
        cov = (2.0 * best.cost / dof) * np.linalg.pinv(best.jac.T @ best.jac)
        std[free] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    tau_at_bound = bool(2 in free and (abs(x[2] - lo[2]) < 1e-3 or abs(x[2] - hi[2]) < 1e-3))
    return dict(R0=float(x[0]), Rp=float(x[1]), tau=float(np.exp(x[2])), drift=float(x[3]), c0=float(x[4]),
                R0_std=float(std[0]), Rp_std=float(std[1]), tau_std=float(np.exp(x[2]) * std[2]),
                c0_std=float(std[4]), rmse_mV=1e3 * rmse(best.fun), n=int(w.sum()), tau_at_bound=tau_at_bound,
                vhat=pulse_predict(x, t, I, dsoc))


def analyse_pulse(cell, p, soc_fn, imp_row=None, fixed=None):
    """Fit one step and compare (V(t_on + d) - V_before) / dI, measured and modelled, with the file."""
    t, I, V, ds, w = pulse_window(cell, p, soc_fn)
    fixed = dict(fixed or {})
    if p["kind"] == "rest":
        fixed["c0"] = p["v_pre"]                  # measured rest voltage = OCV point
    f = fit_pulse(t, I, V, ds, w, c0_init=p["v_pre"] - p["I_prev"] * 0.04, fixed=fixed)
    f.update(p=p, kind=p["kind"], soc=float(soc_fn(np.array([p["start"]]))[0]))
    d_i = p["I"] - p["I_prev"]
    pre = np.flatnonzero((t < 0) & w)
    on = np.flatnonzero((t > 0) & (t <= p["dur"] + 1e-9) & w)
    for d, nm in ((1.0, "R1s"), (30.0, "R30s")):
        ok = pre.size and on.size >= 2 and t[on[0]] - 0.5 * p["dt_med"] <= d <= t[on[-1]] + p["dt_med"] + 1e-9
        f[nm + "_direct"] = float((np.interp(d, t[on], V[on]) - V[pre[-1]]) / d_i) if ok else np.nan
        f[nm + "_model"] = float((np.interp(d, t[on], f["vhat"][on]) - f["vhat"][pre[-1]]) / d_i) if ok else np.nan
    f["R1s_file"], f["R30s_file"] = impedance_for_soc(imp_row, f["soc"])
    f["ok"] = bool(f["n"] >= 8 and f["rmse_mV"] <= 1e3 * PULSE_MAX_RMSE_V and not f["tau_at_bound"])
    return f


def load_impedance(path):
    df = pd.read_csv(path)
    cyc = next((c for c in df.columns if "cycle" in c.lower()), None)
    if cyc is not None:
        df = df.sort_values(cyc).reset_index(drop=True)
    return df, cyc


def impedance_label_socs():
    """SoC of the file's '20%' and '60%' columns. [DATA] On VAH01 these are depth of discharge: the RPT steps
    sit at SoC ~0.80 and ~0.41 at BOL, and dV/dI there reproduces the file's values."""
    return [((1.0 - lab / 100.0) if IMPEDANCE_LABEL == "dod" else lab / 100.0, lab) for lab in (20, 60)]


def impedance_for_soc(imp_row, soc):
    """R_1s, R_30s from the file row for the label nearest this SoC (within IMPEDANCE_SOC_TOL)."""
    if imp_row is None:
        return np.nan, np.nan
    for s_lab, lab in impedance_label_socs():
        if abs(soc - s_lab) < IMPEDANCE_SOC_TOL:
            return (float(imp_row.get(f"{lab}%_1_second", np.nan)), float(imp_row.get(f"{lab}%_30_second", np.nan)))
    return np.nan, np.nan


def soc_table(rows, key):
    """Average a pulse quantity over pulses that sit at (nearly) the same SoC."""
    if not rows:
        return np.array([]), np.array([])
    rows = sorted(rows, key=lambda r: r["soc"])
    groups = [[rows[0]]]
    for r in rows[1:]:
        if r["soc"] - groups[-1][-1]["soc"] > PULSE_SOC_CLUSTER:
            groups.append([r])
        else:
            groups[-1].append(r)
    xs = np.array([np.mean([r["soc"] for r in g]) for g in groups])
    ys = np.array([np.mean([key(r) for r in g]) for g in groups])
    return xs, ys


def interp_const(x, xs, ys):
    """Linear between points, constant beyond the ends (np.interp's behaviour); one point = constant."""
    x = np.asarray(x, float)
    return np.full(x.shape, ys[0]) if len(xs) == 1 else np.interp(x, xs, ys)


# ============================================================================
# 5. Two-rate long-time resistance (pseudocode step 3)
# ============================================================================

def cc_curve_on_grid(cell, s, e, soc, trim_s, grid):
    t = cell["t"][s:e]
    keep = (t - t[0]) >= trim_s
    return interp_on_grid(soc[keep], cell["V"][s:e][keep], grid)


def two_rate_resistance(cell, blk, fit_events, q, trim_s, grid):
    s, e = blk["cc_start"], blk["cc_stop"]
    v_half = cc_curve_on_grid(cell, s, e, block_soc(cell, blk, np.arange(s, e)), trim_s, grid)
    curves, i_each, t_each = [], [], []
    for ev in fit_events:
        curves.append(cc_curve_on_grid(cell, ev["start"], ev["stop"], event_soc(cell, ev, ev["soc0"], q),
                                       trim_s, grid))
        i_each.append(ev["I"])
        t_each.append(ev["T"])
    if not curves:
        return None
    M, i_each = np.vstack(curves), np.array(i_each)
    count = np.isfinite(M).sum(axis=0)
    v_1c = nanmedian_quiet(M, axis=0)
    v_1c[count < MIN_EVENTS_PER_BIN] = np.nan
    di = i_each.mean() - blk["I"]
    r_raw = (v_1c - v_half) / di
    return dict(v_half=v_half, v_1c=v_1c, r_raw=r_raw, r_smooth=nan_moving_average(r_raw, RCC_SMOOTH_BINS),
                per_event=(M - v_half) / (i_each[:, None] - blk["I"]), curves=M, i_each=i_each,
                count=count, i_half=blk["I"], i_1c=float(i_each.mean()), T_half=blk["T"],
                T_1c=float(np.mean(t_each)))


def bootstrap_rcc(tr, rng, n_boot):
    """Resample the fit events with replacement and redo R_cc(SoC). Returns (n_boot, grid) array."""
    M, i_each = tr["curves"], tr["i_each"]
    out = np.full((n_boot, M.shape[1]), np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(M), len(M))
        med = nanmedian_quiet(M[pick], axis=0)
        med[np.isfinite(M[pick]).sum(axis=0) < MIN_EVENTS_PER_BIN] = np.nan
        out[b] = nan_moving_average((med - tr["v_half"]) / (i_each[pick].mean() - tr["i_half"]), RCC_SMOOTH_BINS)
    return out


def assemble_resistance(grid, good, tr, imp_row):
    """R_total(SoC), phi(SoC). R_total = measured R_cc where the two rates overlap; outside that,
    the pulse-based shape is attached to the overlap edge (flagged as extrapolated)."""
    info = {}
    if good:
        xs, r0s = soc_table(good, lambda r: r["R0"])
        _, rts = soc_table(good, lambda r: r["R0"] + r["Rp"])
        r0_g, rpul_g = interp_const(grid, xs, r0s), interp_const(grid, xs, rts)
        pulse_socs = xs
    elif imp_row is not None:
        # [ASSUMED] fallback: R(1 s)/R(30 s) as an upper-bound proxy for phi, R(30 s) as the total
        labs = sorted(impedance_label_socs())
        xs = np.array([s_lab for s_lab, _ in labs])
        r1 = np.array([float(imp_row.get(f"{lab}%_1_second", np.nan)) for _, lab in labs])
        r30 = np.array([float(imp_row.get(f"{lab}%_30_second", np.nan)) for _, lab in labs])
        r0_g, rpul_g = interp_const(grid, xs, r1), interp_const(grid, xs, r30)
        pulse_socs = xs
        info["fallback"] = "no usable pulse fit: phi from impedance-file R(1s)/R(30s) (upper bound)"
    else:
        raise RuntimeError("No usable pulse fits and no impedance file: cannot split R0/Rp. "
                           "Run --inventory-only and adjust the PULSE_* settings.")

    measured = np.zeros(len(grid), bool)
    if tr is not None:
        measured = np.isfinite(tr["r_smooth"])
    if measured.any():
        lo_i, hi_i = np.flatnonzero(measured)[[0, -1]]
        r_total = np.where(measured, tr["r_smooth"], np.nan)
        r_total[:lo_i] = r_total[lo_i] * rpul_g[:lo_i] / rpul_g[lo_i]
        r_total[hi_i + 1:] = r_total[hi_i] * rpul_g[hi_i + 1:] / rpul_g[hi_i]
        inside = [s for s in pulse_socs if grid[lo_i] <= s <= grid[hi_i]]
        at = np.array(inside) if inside else np.array([grid[lo_i] if abs(pulse_socs[0] - grid[lo_i])
                                                       < abs(pulse_socs[-1] - grid[hi_i]) else grid[hi_i]])
        ratio = float(np.median(np.interp(at, grid, r_total) / np.interp(at, grid, rpul_g)))
        info.update(ratio_rcc_over_pulse=ratio, ratio_at_soc=at.tolist(), ratio_from_overlap=bool(inside),
                    rcc_soc_range=[float(grid[lo_i]), float(grid[hi_i])])
    else:
        r_total, ratio = rpul_g.copy(), 1.0
        info.update(ratio_rcc_over_pulse=np.nan, note="no two-rate overlap: R_total from pulses only")
    info["mode"] = "pulse" if abs(ratio - 1.0) <= RCC_MATCH_TOL or not np.isfinite(ratio) else "effective"
    phi = np.clip(r0_g / r_total, 0.02, 0.98)
    return r_total, phi, r0_g, rpul_g, measured, info


# ============================================================================
# 6. OCV on the charge branch (pseudocode step 4)
# ============================================================================

def build_ocv(cell, blk, r_total_fn, trim_s, grid):
    s, e = blk["cc_start"], blk["cc_stop"]
    idx = np.arange(s, e)
    t = cell["t"][idx]
    keep = (t - t[0]) >= trim_s
    soc = block_soc(cell, blk, idx)[keep]
    ocv_raw = cell["V"][idx][keep] - cell["I"][idx][keep] * r_total_fn(soc)
    lo, hi = float(soc.min()), float(soc.max())
    knots = np.linspace(lo, hi, OCV_KNOTS)
    vals = np.linalg.lstsq(hat_basis(soc, knots), ocv_raw, rcond=None)[0]
    mono = pava_increasing(vals)
    n_fixed = int(np.sum(np.abs(mono - vals) > 1e-9))
    mono = mono + 1e-6 * np.arange(len(mono))          # strictly increasing, so OCV is invertible
    return dict(knots=knots, vals=mono, n_pava_fixed=n_fixed, soc_raw=soc, ocv_raw=ocv_raw,
                valid=(lo, hi), grid_vals=ocv_from_knots(knots, mono, grid))


def ocv_from_knots(knots, vals, grid):
    """PCHIP inside the knot range (monotone for monotone data), linear extrapolation outside."""
    p = PchipInterpolator(knots, vals, extrapolate=False)
    d = p.derivative()
    out = p(grid)
    lo, hi = grid < knots[0], grid > knots[-1]
    out[lo] = vals[0] + d(knots[0]) * (grid[lo] - knots[0])
    out[hi] = vals[-1] + d(knots[-1]) * (grid[hi] - knots[-1])
    return out


# ============================================================================
# 7. Optional joint refinement (pseudocode step 5)
# ============================================================================

def joint_refine(cell, blk, model, ocv_info, good, soc_fn):
    """min  sum(resid^2)/N/rms0^2 + lambda*|D2 theta|^2 (normalised) + mu*|theta - theta0|^2 (scaled).
    Data: the C/2 CC charge (incl. its start transient) and the pulse windows. The pulse windows keep
    their own c0 + drift form, so the pulse voltages stay independent for OCV validation (tier C)."""
    grid = model.grid
    knots = ocv_info["knots"]
    K = len(knots)
    lo, hi = model.soc_valid
    rk = np.linspace(lo, hi, R_KNOTS)
    th0 = np.r_[model.ocv_at(knots), np.log(np.interp(rk, grid, model.r_total)), np.log(np.median(model.tau))]
    s, e = blk["cc_start"], blk["cc_stop"]
    idx = np.arange(s, e)
    tc, Ic, Vc, sc = cell["t"][idx] - cell["t"][s], cell["I"][idx], cell["V"][idx], soc_fn(idx)
    windows = []
    for r in good:
        tt, II, VV, ds, ww = pulse_window(cell, r["p"], soc_fn)
        windows.append((tt, II, VV, ds, ww, r["c0"], r["soc"], r["drift"]))

    def build(th):
        ocv_g = ocv_from_knots(knots, np.sort(th[:K]), grid)
        rt_g = np.exp(np.interp(grid, rk, th[K:K + R_KNOTS]))
        return EcmModel(grid, ocv_g, rt_g, model.phi, np.exp(th[-1]), model.soc_valid, model.q_ah, model.t_ref)

    def data_resid(th):
        m = build(th)
        parts = [m.simulate(tc, Ic, sc) - Vc]
        for tt, II, VV, ds, ww, c0, sp, drift in windows:
            rt = float(np.interp(sp, grid, m.r_total))
            ph = float(np.interp(sp, grid, m.phi))
            parts.append((pulse_predict([ph * rt, (1 - ph) * rt, th[-1], drift, c0], tt, II, ds) - VV)[ww])
        return np.concatenate(parts)

    r_init = data_resid(th0)
    norm = rmse(r_init) * np.sqrt(len(r_init))
    sc_o = max(rmse(np.diff(th0[:K], 2)), 1e-4)
    sc_r = max(rmse(np.diff(th0[K:K + R_KNOTS], 2)), 1e-3)
    prior_scale = np.r_[np.full(K, 0.01), np.full(R_KNOTS, 0.1), 0.3]   # [ASSUMED] 10 mV, 10%, 30%

    def resid(th):
        return np.r_[data_resid(th) / norm,
                     np.sqrt(SMOOTHNESS_WEIGHT) * np.diff(th[:K], 2) / sc_o,
                     np.sqrt(SMOOTHNESS_WEIGHT) * np.diff(th[K:K + R_KNOTS], 2) / sc_r,
                     np.sqrt(PRIOR_WEIGHT) * (th - th0) / prior_scale]

    res = least_squares(resid, th0, max_nfev=200)
    new = build(res.x)
    return new, dict(rmse_before_mV=1e3 * rmse(r_init), rmse_after_mV=1e3 * rmse(data_resid(res.x)),
                     nfev=int(res.nfev), success=bool(res.success))


# ============================================================================
# 8. Validation (tiers A-E)                                                   [DEV-ONLY]
# ============================================================================

def predict_cv_time(model, t, I, soc, vh, v_cv, q):
    """Time (from CC start) at which the predicted voltage first reaches the CV setpoint."""
    hit = np.flatnonzero(vh >= v_cv)
    if hit.size:
        k = int(hit[0])
        if k == 0:
            return 0.0
        return float(t[k - 1] + (v_cv - vh[k - 1]) * (t[k] - t[k - 1]) / (vh[k] - vh[k - 1]))
    i_m = float(I.mean())                          # continue at the same current past the data
    t_ext = t[-1] + np.arange(1.0, CV_EXTEND_MAX_S + 1.0)
    vv = model.simulate(np.r_[t, t_ext], np.r_[I, np.full(t_ext.size, i_m)],
                        np.r_[soc, soc[-1] + i_m * (t_ext - t[-1]) / 3600.0 / q])[len(t):]
    hit = np.flatnonzero(vv >= v_cv)
    return float(t_ext[hit[0]]) if hit.size else np.nan


def simulate_event(cell, model, ev, soc0, q, **kw):
    s, e = ev["start"], ev["stop"]
    t = cell["t"][s:e] - cell["t"][s]
    I, V = cell["I"][s:e], cell["V"][s:e]
    soc = event_soc(cell, ev, soc0, q)
    return t, I, V, soc, model.simulate(t, I, soc, **kw)


def validate_events(cell, model, evs, q):
    """Tier B: held-out mission 1C charges with known-duty-cycle SoC0 and oracle coulomb counting."""
    rows, pooled = [], []
    for ev in evs:
        t, I, V, soc, vh = simulate_event(cell, model, ev, ev["soc0"], q)
        res = vh - V
        v_cv = ev["v_cv"] if np.isfinite(ev["v_cv"]) else V_CV_DEFAULT
        t_cv = predict_cv_time(model, t, I, soc, vh, v_cv, q)
        rows.append(dict(event=ev["k"], soc0=ev["soc0"], soc0_bwd_minus_fwd=soc0_backward(cell, ev, q) - ev["soc0"],
                         rmse_mV=1e3 * rmse(res), max_abs_mV=1e3 * float(np.max(np.abs(res))),
                         end_of_cc_err_mV=1e3 * float(res[-1]), cc_dur_s=float(t[-1]),
                         cv_time_err_s=t_cv - float(t[-1]), T_mean=ev["T"]))
        pooled.append(np.c_[soc, res, cell["T"][ev["start"]:ev["stop"]], t])
    return pd.DataFrame(rows), (np.vstack(pooled) if pooled else np.empty((0, 4)))


def sensitivity(cell, model, evs, q):
    """Tier E: 0-RC baseline and how much RMSE moves per unit error in SoC0, R_total and phi."""
    cases = [("1RC baseline", 0.0, {}), ("0-RC: OCV + I*R_total", 0.0, dict(zero_rc=True))]
    for d in SOC0_PERTURB:
        cases += [(f"SoC0 {s * d * 100:+.0f}%", s * d, {}) for s in (+1, -1)]
    cases += [(f"R_total {s * RTOT_PERTURB * 100:+.0f}%", 0.0, dict(r_scale=1 + s * RTOT_PERTURB)) for s in (+1, -1)]
    cases += [(f"phi {s * PHI_PERTURB:+.2f}", 0.0, dict(phi_shift=s * PHI_PERTURB)) for s in (+1, -1)]
    rows = []
    for name, dsoc0, kw in cases:
        errs = [rmse(simulate_event(cell, model, ev, ev["soc0"] + dsoc0, q, **kw)[4]
                     - cell["V"][ev["start"]:ev["stop"]]) for ev in evs]
        rows.append(dict(case=name, mean_rmse_mV=1e3 * float(np.mean(errs))))
    df = pd.DataFrame(rows)
    df["delta_vs_baseline_mV"] = df["mean_rmse_mV"] - df["mean_rmse_mV"].iloc[0]
    return df


def aging_floor(cell, model, events, fulls, q_of_t, q_bol, stride):
    """Tier D: per charge, fit ONE scalar s(c) on R_total, keep BOL OCV and phi, Q(c) from the RPTs.
    The remaining error is what no resistance estimator can remove with a fixed OCV/split."""
    rows = []
    use = [ev for ev in events if not ev["after_rpt"]][::max(1, stride)]
    for ev in use:
        V = cell["V"][ev["start"]:ev["stop"]]
        q_c = float(q_of_t(ev["t0"]))
        soc0_c = soc0_forward(cell, fulls, ev["start"], q_c)
        soc0_b = soc0_forward(cell, fulls, ev["start"], q_bol)
        if not (np.isfinite(soc0_c) and np.isfinite(soc0_b)):
            continue
        t, I, _, soc, _ = simulate_event(cell, model, ev, soc0_c, q_c)

        def f(z):
            return model.simulate(t, I, soc, r_scale=z[0]) - V

        def f2(z):
            return model.simulate(t, I, soc + z[1], r_scale=z[0]) - V

        fit = least_squares(f, [1.0], bounds=([0.2], [5.0]))
        # Variant: also free a SoC offset. If this needs s < 1 and a growing offset, the floor error is the
        # frozen OCV-vs-SoC map drifting with age, not resistance.
        fit2 = least_squares(f2, [1.0, 0.0], bounds=([0.2, -0.2], [5.0, 0.2]))
        rows.append(dict(event=ev["k"], t0_h=ev["t0"] / 3600.0, q_c=q_c, soc0=soc0_c, scale=float(fit.x[0]),
                         rmse_floor_mV=1e3 * rmse(fit.fun), scale_with_soc_offset=float(fit2.x[0]),
                         soc_offset=float(fit2.x[1]), rmse_floor_soc_offset_mV=1e3 * rmse(fit2.fun),
                         rmse_bol_mV=1e3 * rmse(simulate_event(cell, model, ev, soc0_b, q_bol)[4] - V)))
    return pd.DataFrame(rows)


def rpt_pulse_series(cell, model, blocks, imp_df=None, imp_cyc=None):
    """Tier C plus aging evidence. At every RPT, fit each step with that RPT's own Q and SoC anchor:
      - OCV check: the step's OCV level (measured V_pre for 'rest' steps; fitted c0 for 'step' steps, which is
        a discharge-branch value) against the BOL charge-branch OCV. Never used in fitting.
      - R0, Rp, tau, phi per RPT: direct evidence for a fixed vs aging-dependent split (Ablations A/B)."""
    rows = []
    for i, b in enumerate(blocks):
        imp_row = None
        if imp_df is not None and imp_cyc is not None and len(imp_df):
            imp_row = imp_df.iloc[int(np.argmin(np.abs(imp_df[imp_cyc].to_numpy(float) - (b["cycle"] + 1))))].to_dict()

        def soc_fn(idx, b=b):
            return block_soc(cell, b, idx)

        for p in find_pulses(cell, b["full"], b["b1"]):
            f = analyse_pulse(cell, p, soc_fn, imp_row)
            ocv = float(model.ocv_at(f["soc"])[0])
            rows.append(dict(rpt=i, cycle=b["cycle"], q_Ah=b["q"], kind=p["kind"], soc=f["soc"],
                             ocv_level=f["c0"], ocv_level_source="measured rest V" if p["kind"] == "rest"
                             else "fitted c0 (discharge branch)", ocv_bol=ocv, err_mV=1e3 * (ocv - f["c0"]),
                             in_valid_range=bool(model.soc_valid[0] <= f["soc"] <= model.soc_valid[1]),
                             R0=f["R0"], Rp=f["Rp"], tau=f["tau"], phi=f["R0"] / max(f["R0"] + f["Rp"], 1e-12),
                             rmse_mV=f["rmse_mV"], ok=f["ok"], R1s_direct=f["R1s_direct"], R1s_model=f["R1s_model"],
                             R1s_file=f["R1s_file"], R30s_direct=f["R30s_direct"], R30s_model=f["R30s_model"],
                             R30s_file=f["R30s_file"], T=p["T"]))
    return pd.DataFrame(rows)


def discharge_branch_ocv(cell, blk, r_total_fn, trim_s, grid):
    """[DEV-ONLY diagnostic] Low-rate discharge legs of the RPT (VAH01: C/5 from full to 2.5 V), corrected
    with the charge-measured R_total: OCV_dis = V - I*R_total. The gap to the charge-branch OCV estimates
    hysteresis plus the error of using charge resistance on discharge. Not used in the hand-off."""
    t, I, V, ns = cell["t"], cell["I"], cell["V"], cell["ns"]
    socs, vals = [], []
    for s, e in [(a + blk["full"], c + blk["full"]) for a, c in funcruns(ns[blk["full"]:blk["b1"]])]:
        if np.median(I[s:e]) < -REST_CURRENT_A and seg_duration(t, s, e) >= 600.0:
            keep = np.arange(s, e)[(t[s:e] - t[s]) >= trim_s]
            if keep.size:
                soc = block_soc(cell, blk, keep)
                socs.append(soc)
                vals.append(V[keep] - I[keep] * r_total_fn(soc))
    if not socs:
        return np.full(len(grid), np.nan)
    soc, val = np.concatenate(socs), np.concatenate(vals)
    k = np.clip(np.rint(soc * (len(grid) - 1)).astype(int), 0, len(grid) - 1)     # bin to the nearest grid point
    cnt = np.bincount(k, minlength=len(grid))
    out = np.full(len(grid), np.nan)
    out[cnt > 0] = np.bincount(k, weights=val, minlength=len(grid))[cnt > 0] / cnt[cnt > 0]
    return out


# ============================================================================
# 9. Plots
# ============================================================================

def _style(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def plot_pulses(cell, fits, path):
    if not fits:
        return
    n = min(len(fits), 6)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 3.4), squeeze=False)
    for ax, r in zip(axs[0], fits[:n]):
        p = r["p"]
        t = cell["t"][p["win0"]:p["win1"]] - cell["t"][p["start"]]
        ax.plot(t, cell["V"][p["win0"]:p["win1"]], ".", ms=3, label="measured")
        ax.plot(t, r["vhat"], "-", lw=1.2, label="1RC fit")
        ax.set_xlim(-60, p["dur"] + 120)
        _style(ax, "time from step start [s]", "V [V]",
               f"{r['kind']} at SoC {r['soc']:.2f}, I {p['I_prev']:+.2f} -> {p['I']:+.2f} A\nR0 {1e3 * r['R0']:.1f}, Rp {1e3 * r['Rp']:.1f} mOhm, "
               f"tau {r['tau']:.0f} s")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ecm(model, ocv_info, tr, fits, ocv_df, imp_row, boot, ocv_dis, path):
    g = model.grid
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    ax = axs[0, 0]
    ax.plot(ocv_info["soc_raw"], ocv_info["ocv_raw"], ".", ms=2, alpha=0.4, label="V - I*R_total (C/2 charge)")
    ax.plot(g, model.ocv, "k-", lw=1.2, label="OCV fit (charge branch)")
    if np.isfinite(ocv_dis).any():
        ax.plot(g, ocv_dis, "-", color="tab:blue", lw=0.8, label="discharge branch (diagnostic)")
    if len(ocv_df):
        bol = ocv_df[ocv_df["rpt"] == 0]
        later = ocv_df[ocv_df["rpt"] > 0]
        ax.plot(bol["soc"], bol["ocv_level"], "rs", ms=6, label="BOL step OCV level (not fitted)")
        if len(later):
            ax.plot(later["soc"], later["ocv_level"], "x", color="tab:orange", ms=5, label="later-RPT step level")
    ax.axvspan(*model.soc_valid, color="tab:green", alpha=0.06)
    _style(ax, "SoC", "V", "OCV, charge branch (shaded = fitted range)")
    ax.legend(fontsize=7)

    ax = axs[0, 1]
    ax.plot(g, 1e3 * np.gradient(model.ocv, g) / 100.0, "k-")
    ax.axvspan(*model.soc_valid, color="tab:green", alpha=0.06)
    _style(ax, "SoC", "dOCV/dSoC [mV per %]", "OCV slope")

    ax = axs[1, 0]
    if tr is not None:
        ax.plot(g, 1e3 * tr["r_raw"], ".", ms=2, color="0.6", label="R_cc raw (two-rate)")
    ax.plot(g, 1e3 * model.r_total, "k-", label="R_total used")
    if boot is not None:
        ax.fill_between(g, 1e3 * boot[0], 1e3 * boot[1], color="k", alpha=0.12, label="bootstrap 5-95%")
    ax.plot(g, 1e3 * model.phi * model.r_total, "b--", label="R0 = phi*R_total")
    for r in fits:
        ax.plot(r["soc"], 1e3 * r["R0"], "bo", ms=5)
        ax.plot(r["soc"], 1e3 * (r["R0"] + r["Rp"]), "ro", ms=5)
    if imp_row is not None:
        for s_lab, lab in impedance_label_socs():
            ax.plot(s_lab, 1e3 * float(imp_row.get(f"{lab}%_1_second", np.nan)), "b^", ms=7)
            ax.plot(s_lab, 1e3 * float(imp_row.get(f"{lab}%_30_second", np.nan)), "r^", ms=7)
    _style(ax, "SoC", "mOhm", "Resistance (o: pulse R0 / R0+Rp, ^: file R(1s) / R(30s))")
    ax.legend(fontsize=7)

    ax = axs[1, 1]
    ax.plot(g, model.phi, "k-")
    for r in fits:
        ax.plot(r["soc"], r["R0"] / (r["R0"] + r["Rp"]), "o", color="tab:purple", ms=5)
    ax.set_ylim(0, 1)
    _style(ax, "SoC", "phi = R0 / R_total", "Ohmic fraction (o: per pulse, against its own R0+Rp)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_validation(cell, model, val_evs, q, pooled_b, half_res, path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ev in val_evs[:3]:
        t, I, V, soc, vh = simulate_event(cell, model, ev, ev["soc0"], q)
        l, = axs[0].plot(t / 60, V, ".", ms=3)
        axs[0].plot(t / 60, vh, "-", color=l.get_color(), lw=1)
    _style(axs[0], "minutes into CC", "V", "Tier B: held-out 1C charges (dots measured, lines model)")
    if len(half_res):
        axs[1].plot(half_res[:, 0], 1e3 * half_res[:, 1], ".", ms=2, color="0.5", label="C/2 (tier A, in-sample)")
    if len(pooled_b):
        axs[1].plot(pooled_b[:, 0], 1e3 * pooled_b[:, 1], ".", ms=3, color="tab:red", label="1C held-out (tier B)")
    axs[1].axhline(0, color="k", lw=0.8)
    _style(axs[1], "SoC", "model - measured [mV]", "Residual vs SoC")
    axs[1].legend(fontsize=7)
    if len(pooled_b):
        axs[2].plot(pooled_b[:, 2] - model.t_ref, 1e3 * pooled_b[:, 1], ".", ms=3, color="tab:red")
    _style(axs[2], "T - T_ref [K]", "model - measured [mV]", "Residual vs temperature (tier B)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rpt_series(df, path):
    """Per-RPT step fits vs cycle: does the split stay fixed (Ablation A) or drift (Ablation B)?"""
    if df is None or not len(df):
        return
    df = df[df["ok"]]
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
    for lvl, grp in df.groupby(np.where(df["soc"] > 0.6, "upper step", "lower step")):
        g = grp.sort_values("cycle")
        axs[0].plot(g["cycle"], 1e3 * g["R0"], "o-", ms=3, label=f"{lvl} R0")
        axs[0].plot(g["cycle"], 1e3 * (g["R0"] + g["Rp"]), "s--", ms=3, label=f"{lvl} R0+Rp")
        axs[1].plot(g["cycle"], g["phi"], "o-", ms=3, label=f"{lvl} fit")
        if g["R1s_file"].notna().any():
            axs[1].plot(g["cycle"], g["R1s_file"] / g["R30s_file"], "x:", ms=4, label=f"{lvl} file R1s/R30s")
        axs[2].plot(g["cycle"], g["tau"], "o-", ms=3, label=lvl)
        axs[3].plot(g["cycle"], g["err_mV"], "o-", ms=3, label=lvl)
    _style(axs[0], "cycle", "mOhm", "Step-fit resistances per RPT")
    _style(axs[1], "cycle", "phi", "Ohmic fraction per RPT")
    _style(axs[2], "cycle", "tau [s]", "tau per RPT")
    _style(axs[3], "cycle", "BOL OCV - step level [mV]", "OCV drift / hysteresis per RPT")
    for ax in axs:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_aging(df, path):
    if df is None or not len(df):
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(df["event"], df["scale"], ".-")
    _style(axs[0], "mission charge (event index)", "s(c) = R_total / R_total_BOL", "Tier D: fitted resistance scale")
    axs[1].plot(df["event"], df["rmse_bol_mV"], ".-", label="frozen BOL model (s = 1, Q_BOL)")
    axs[1].plot(df["event"], df["rmse_floor_mV"], ".-", label="floor: best s(c), Q(c), BOL OCV and phi")
    axs[1].plot(df["event"], df["rmse_floor_soc_offset_mV"], ".-", label="floor + free SoC offset")
    _style(axs[1], "mission charge (event index)", "RMSE [mV]", "Tier D: aging floor")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================================
# 10. Main
# ============================================================================

def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main(argv=None):
    global N_FIT_EVENTS, N_VAL_EVENTS, AGING_STRIDE, RUN_JOINT_REFINE, MAKE_PLOTS, CC_CURRENT_A, RPT_CURRENT_A
    ap = argparse.ArgumentParser(description="Step 2: BOL ECM calibration from measured V/I.")
    ap.add_argument("--csv", required=True, help="cell time-series CSV (e.g. VAH01.csv)")
    ap.add_argument("--impedance", default=None, help="optional <cell>_impedance.csv, for cross-checks")
    ap.add_argument("--out", default="step2_results")
    ap.add_argument("--inventory-only", action="store_true", help="print what is in the BOL block and stop")
    ap.add_argument("--bol-rpt", type=int, default=0, help="which RPT block to calibrate on (default 0)")
    ap.add_argument("--n-fit-events", type=int, default=N_FIT_EVENTS)
    ap.add_argument("--n-val-events", type=int, default=N_VAL_EVENTS)
    ap.add_argument("--aging-stride", type=int, default=AGING_STRIDE, help="tier D: every Nth charge (0 = skip)")
    ap.add_argument("--cc-current", type=float, default=CC_CURRENT_A)
    ap.add_argument("--rpt-current", type=float, default=RPT_CURRENT_A)
    ap.add_argument("--joint-refine", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    N_FIT_EVENTS, N_VAL_EVENTS, AGING_STRIDE = args.n_fit_events, args.n_val_events, args.aging_stride
    CC_CURRENT_A, RPT_CURRENT_A = args.cc_current, args.rpt_current
    RUN_JOINT_REFINE = RUN_JOINT_REFINE or args.joint_refine
    MAKE_PLOTS = MAKE_PLOTS and not args.no_plots
    t_start = time.time()

    # ---- 0. inventory --------------------------------------------------------------------------
    cell = load_cell(args.csv)
    events = find_mission_events(cell)
    blocks = find_rpt_blocks(cell, events)
    if args.inventory_only:
        print_inventory(cell, events, blocks)
        return 0
    if not blocks:
        print_inventory(cell, events, blocks)
        sys.exit("No RPT C/2 charge found. Check RPT_CURRENT_A / RPT_MIN_DURATION_S.")
    os.makedirs(args.out, exist_ok=True)
    name = cell["name"]
    blk = blocks[args.bol_rpt]
    imp_row, imp_df, imp_cyc = None, None, None
    if args.impedance:
        imp_df, imp_cyc = load_impedance(args.impedance)
        imp_row = imp_df.iloc[0].to_dict()                     # BOL row = lowest cycle number

    # ---- 1. coulomb counting -----------------------------------------------------------------------
    q_bol = blk["q"]
    fulls = full_indices(events, blocks)
    for ev in events:
        ev["soc0"] = soc0_forward(cell, fulls, ev["start"], q_bol)
    nxt_rpt = blocks[args.bol_rpt + 1]["full"] if args.bol_rpt + 1 < len(blocks) else cell["n"]
    bol_ev = [ev for ev in events if blk["full"] < ev["start"] < nxt_rpt and np.isfinite(ev["soc0"])
              and (USE_POST_RPT_1C or not ev["after_rpt"])]
    fit_ev, val_ev = bol_ev[:N_FIT_EVENTS], bol_ev[N_FIT_EVENTS:N_FIT_EVENTS + N_VAL_EVENTS]
    print(f"{name}: {len(events)} 1C charges, {len(blocks)} RPTs. BOL RPT {args.bol_rpt}: Q_BOL = {q_bol:.4f} Ah "
          f"({blk['q_src']}), C/2 charge {blk['I']:.3f} A. 1C charges for R_cc: {len(fit_ev)}, held out: {len(val_ev)}")
    if len(val_ev) == 0:
        print("  WARNING: no held-out 1C charges; tiers B and E will be empty.")

    # ---- 2. pulse fits ---------------------------------------------------------------------------
    def soc_fn(idx):
        return block_soc(cell, blk, idx)

    fits = [analyse_pulse(cell, p, soc_fn, imp_row) for p in find_pulses(cell, blk["full"], blk["b1"])]
    good = [f for f in fits if f["ok"]]
    print(f"  pulses: {len(fits)} found, {len(good)} usable")
    for f in fits:
        print(f"    {f['kind']} SoC {f['soc']:.3f} I {f['p']['I_prev']:+.2f}->{f['p']['I']:+.2f} A: R0 {1e3 * f['R0']:.2f}  Rp {1e3 * f['Rp']:.2f} mOhm  "
              f"tau {f['tau']:.1f} s  drift {f['drift'] * 10:.1f} mV/%  rmse {f['rmse_mV']:.2f} mV"
              f"{'' if f['ok'] else '  [NOT USED]'}  | R(1s) file {1e3 * f['R1s_file']:.1f} / direct "
              f"{1e3 * f['R1s_direct']:.1f} / model {1e3 * f['R1s_model']:.1f};  R(30s) file "
              f"{1e3 * f['R30s_file']:.1f} / direct {1e3 * f['R30s_direct']:.1f} / model {1e3 * f['R30s_model']:.1f}")
    if fits and not good:
        print("  WARNING: no usable pulse fit (check sampling: tau at a bound usually means it is too coarse).")
    tau_pulse = float(np.median([f["tau"] for f in good])) if good else TAU_FALLBACK_S
    trim_s = max(TRIM_MIN_S, TRIM_TAU_MULT * tau_pulse)

    # ---- 3. two-rate long-time resistance ---------------------------------------------------------
    grid = np.linspace(0.0, 1.0, SOC_GRID_N)
    tr = two_rate_resistance(cell, blk, fit_ev, q_bol, max(trim_s, RCC_TRIM_S), grid)
    r_total, phi, r0_g, rpul_g, measured, rinfo = assemble_resistance(grid, good, tr, imp_row)
    if tr is not None:
        print(f"  R_cc over SoC {rinfo.get('rcc_soc_range')}: R_cc / R_pulse = {rinfo['ratio_rcc_over_pulse']:.3f} "
              f"at SoC {np.round(rinfo['ratio_at_soc'], 3).tolist()} -> mode '{rinfo['mode']}'. "
              f"T during C/2 {tr['T_half']:.1f} C vs 1C {tr['T_1c']:.1f} C")

    # tau: pulse value, or an effective tau with Rp forced to R_total - R0 ("effective" mode)
    if rinfo["mode"] == "effective" and good:
        taus = []
        for f in good:
            rt = float(np.interp(f["soc"], grid, r_total))
            g2 = analyse_pulse(cell, f["p"], soc_fn, fixed=dict(R0=f["R0"], Rp=max(rt - f["R0"], 0.0)))
            f["tau_eff"] = g2["tau"]
            taus.append(g2["tau"])
        tau_used = float(np.median(taus))
        print(f"  effective tau (Rp = R_cc - R0, refit on pulses): {tau_used:.1f} s (pulse tau {tau_pulse:.1f} s)")
    else:
        tau_used = tau_pulse
    tau_grid = tau_used
    if TAU_PER_SOC and len(good) > 1:
        xs, ts = soc_table(good, lambda r: r.get("tau_eff", r["tau"]))
        tau_grid = interp_const(grid, xs, ts)

    # ---- 4. OCV ------------------------------------------------------------------------------------
    def r_total_fn(s):
        return np.interp(s, grid, r_total)

    ocv_info = build_ocv(cell, blk, r_total_fn, trim_s, grid)
    t_ref = float(np.mean([blk["T"]] + [f["p"]["T"] for f in good]))
    model = EcmModel(grid, ocv_info["grid_vals"], r_total, phi, tau_grid, ocv_info["valid"], q_bol, t_ref)
    slope = np.gradient(model.ocv, grid)
    in_rng = (grid >= model.soc_valid[0]) & (grid <= model.soc_valid[1])
    print(f"  OCV fitted on SoC {model.soc_valid[0]:.3f}-{model.soc_valid[1]:.3f}; "
          f"{ocv_info['n_pava_fixed']} knots needed monotone repair; "
          f"slope {slope[in_rng].min() / 100 * 1e3:.2f}-{slope[in_rng].max() / 100 * 1e3:.2f} mV/%")
    ocv_dis = discharge_branch_ocv(cell, blk, r_total_fn, trim_s, grid)
    hyst = np.where(in_rng, model.ocv - ocv_dis, np.nan)
    if np.isfinite(hyst).any():
        print(f"  charge - discharge OCV gap (hysteresis + R error) over the fitted range: median "
              f"{1e3 * np.nanmedian(hyst):.1f} mV, range {1e3 * np.nanmin(hyst):.1f} to {1e3 * np.nanmax(hyst):.1f} mV")

    # ---- 5. optional joint refinement ---------------------------------------------------------------
    refine_info = None
    if RUN_JOINT_REFINE:
        model, refine_info = joint_refine(cell, blk, model, ocv_info, good, soc_fn)
        print(f"  joint refinement: data RMSE {refine_info['rmse_before_mV']:.2f} -> "
              f"{refine_info['rmse_after_mV']:.2f} mV ({refine_info['nfev']} evals)")

    # ---- uncertainty ----------------------------------------------------------------------------------
    boot_r = None
    unc = dict(pulse_param_std=[dict(soc=f["soc"], R0_std=f["R0_std"], Rp_std=f["Rp_std"], tau_std=f["tau_std"])
                                for f in good])
    if tr is not None and BOOTSTRAP_N > 0 and len(fit_ev) > 1:
        B = bootstrap_rcc(tr, np.random.default_rng(RANDOM_SEED), BOOTSTRAP_N)
        boot_r = nanpercentile_quiet(B, [5, 95], axis=0)
        boot_phi = nanpercentile_quiet(r0_g[None, :] / B, [5, 95], axis=0)
        unc.update(r_total_p05=boot_r[0], r_total_p95=boot_r[1], phi_p05=boot_phi[0], phi_p95=boot_phi[1],
                   note="bootstrap over the 1C fit charges; NaN outside the two-rate overlap")

    # ---- validation ------------------------------------------------------------------------------------
    # A: in-sample C/2 charge
    s, e = blk["cc_start"], blk["cc_stop"]
    idx = np.arange(s, e)
    th = cell["t"][idx] - cell["t"][s]
    vh = model.simulate(th, cell["I"][idx], soc_fn(idx))
    res_half = vh - cell["V"][idx]
    half_res = np.c_[soc_fn(idx), res_half]
    tierA = dict(rmse_mV=1e3 * rmse(res_half), rmse_after_trim_mV=1e3 * rmse(res_half[th >= trim_s]),
                 max_abs_mV=1e3 * float(np.max(np.abs(res_half))))
    print(f"  tier A (C/2, in-sample): RMSE {tierA['rmse_mV']:.2f} mV, after trim {tierA['rmse_after_trim_mV']:.2f} mV")

    # B and E: held-out BOL 1C charges
    tierB_df, pooled_b = validate_events(cell, model, val_ev, q_bol)
    tierE_df = sensitivity(cell, model, val_ev, q_bol) if val_ev else pd.DataFrame()
    tierB = {}
    if len(tierB_df):
        r_T = float(np.corrcoef(pooled_b[:, 2], pooled_b[:, 1])[0, 1]) if np.std(pooled_b[:, 2]) > 0 else np.nan
        tierB = dict(mean_rmse_mV=float(tierB_df["rmse_mV"].mean()), max_abs_mV=float(tierB_df["max_abs_mV"].max()),
                     mean_end_of_cc_err_mV=float(tierB_df["end_of_cc_err_mV"].mean()),
                     mean_abs_cv_time_err_s=float(tierB_df["cv_time_err_s"].abs().mean()),
                     n_cv_not_reached=int(tierB_df["cv_time_err_s"].isna().sum()),
                     residual_temperature_corr=r_T, ratio_to_tierA=float(tierB_df["rmse_mV"].mean())
                     / max(tierA["rmse_after_trim_mV"], 1e-9))
        print(f"  tier B (held-out 1C): mean RMSE {tierB['mean_rmse_mV']:.2f} mV ({tierB['ratio_to_tierA']:.1f}x tier A), "
              f"end-of-CC {tierB['mean_end_of_cc_err_mV']:+.2f} mV, |CV timing| {tierB['mean_abs_cv_time_err_s']:.0f} s, "
              f"residual-T corr {r_T:+.2f}")
        print("  tier E (held-out 1C, mean RMSE and change vs baseline):")
        for _, r in tierE_df.iterrows():
            print(f"    {r['case']:<26} {r['mean_rmse_mV']:7.2f} mV  ({r['delta_vs_baseline_mV']:+.2f})")

    # C: OCV against the step OCV levels at every RPT (+ per-RPT R0, Rp, tau, phi for Ablation B)
    ocv_df = rpt_pulse_series(cell, model, blocks, imp_df, imp_cyc)
    if len(ocv_df):
        bol_c = ocv_df[ocv_df["rpt"] == args.bol_rpt]
        print("  tier C (BOL charge OCV - step OCV level): " + ", ".join(
            f"SoC {a:.2f} ({k}): {b:+.1f} mV" for a, b, k in zip(bol_c["soc"], bol_c["err_mV"], bol_c["kind"])))
        ok_df = ocv_df[ocv_df["ok"]]
        for lvl, grp in ok_df.groupby(np.where(ok_df["soc"] > 0.6, "upper", "lower")):
            g = grp.sort_values("cycle")
            print(f"  per-RPT step fits, {lvl} step: phi {g['phi'].iloc[0]:.2f} -> {g['phi'].iloc[-1]:.2f}, "
                  f"R0 x{g['R0'].iloc[-1] / g['R0'].iloc[0]:.2f}, R0+Rp x{(g['R0'] + g['Rp']).iloc[-1] / (g['R0'] + g['Rp']).iloc[0]:.2f}, "
                  f"tau {g['tau'].iloc[0]:.0f} -> {g['tau'].iloc[-1]:.0f} s over cycles {g['cycle'].iloc[0]}-{g['cycle'].iloc[-1]}")

    # D: aging floor across life
    floor_df = None
    if AGING_STRIDE > 0:
        tq = np.array([b["t_full"] for b in blocks])
        qq = np.array([b["q"] for b in blocks])

        def q_of_t(t0):
            return np.interp(t0, tq, qq)

        floor_df = aging_floor(cell, model, events, fulls, q_of_t, q_bol, AGING_STRIDE)
        if len(floor_df):
            last = floor_df.tail(max(1, len(floor_df) // 10))
            first = floor_df.head(max(1, len(floor_df) // 10))
            print(f"  tier D (aging floor, {len(floor_df)} charges): first/last 10% of life: frozen-BOL RMSE "
                  f"{first['rmse_bol_mV'].mean():.1f} -> {last['rmse_bol_mV'].mean():.1f} mV; best s(c): s "
                  f"{first['scale'].mean():.2f} -> {last['scale'].mean():.2f}, RMSE {first['rmse_floor_mV'].mean():.1f} -> "
                  f"{last['rmse_floor_mV'].mean():.1f} mV; with a SoC offset: s {last['scale_with_soc_offset'].mean():.2f}, "
                  f"offset {last['soc_offset'].mean():+.3f}, RMSE {last['rmse_floor_soc_offset_mV'].mean():.1f} mV")

    # ---- 6. outputs ------------------------------------------------------------------------------------
    pulse_cols = ["kind", "soc", "R0", "Rp", "tau", "drift", "c0", "R0_std", "Rp_std", "tau_std", "c0_std", "rmse_mV", "n", "tau_at_bound",
                  "ok", "R1s_direct", "R1s_model", "R1s_file", "R30s_direct", "R30s_model", "R30s_file", "tau_eff"]
    pdf = pd.DataFrame([{c: f.get(c, np.nan) for c in pulse_cols} | dict(I_prev=f["p"]["I_prev"], I=f["p"]["I"],
                         dur_s=f["p"]["dur"], dt_med_s=f["p"]["dt_med"], v_pre=f["p"]["v_pre"], T=f["p"]["T"]) for f in fits])
    pdf.to_csv(os.path.join(args.out, f"{name}_pulses.csv"), index=False)
    grid_df = pd.DataFrame(dict(soc=grid, ocv_V=model.ocv, r_total_ohm=model.r_total, phi=model.phi,
                                r0_ohm=model.phi * model.r_total, rp_ohm=(1 - model.phi) * model.r_total,
                                tau_s=model.tau, rcc_measured=measured,
                                ocv_in_fit_range=(grid >= model.soc_valid[0]) & (grid <= model.soc_valid[1]),
                                rcc_raw_ohm=tr["r_raw"] if tr is not None else np.nan,
                                v_half=tr["v_half"] if tr is not None else np.nan,
                                v_1c=tr["v_1c"] if tr is not None else np.nan,
                                ocv_discharge_branch_V=ocv_dis, ocv_charge_minus_discharge_V=hyst))
    grid_df.to_csv(os.path.join(args.out, f"{name}_ecm_grid.csv"), index=False)
    pd.DataFrame([{k: b[k] for k in ("cycle", "t_full", "q", "q_chg", "q_dis", "q_src", "dis_empty", "chg_empty",
                                     "I", "T")} for b in blocks]).to_csv(
        os.path.join(args.out, f"{name}_rpt_capacity.csv"), index=False)
    tierB_df.to_csv(os.path.join(args.out, f"{name}_tierB_heldout.csv"), index=False)
    tierE_df.to_csv(os.path.join(args.out, f"{name}_tierE_sensitivity.csv"), index=False)
    ocv_df.to_csv(os.path.join(args.out, f"{name}_tierC_rpt_steps.csv"), index=False)
    if floor_df is not None:
        floor_df.to_csv(os.path.join(args.out, f"{name}_tierD_aging_floor.csv"), index=False)

    handoff = dict(cell=name, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                   sign_convention="charge-positive: V = OCV(SoC) + I*R0 + V1; V1+ = a*V1 + Rp*(1-a)*I, a = exp(-dt/tau)",
                   model=model.to_dict(), resistance_info=rinfo, trim_s=trim_s, tau_pulse_s=tau_pulse,
                   uncertainty=unc,
                   usage={"R_total(t)": "r_total_ohm(SoC) * R_ratio_T(t) from Step 1 [Step 3]",
                          "R0, Rp": "phi(SoC) * R_total, (1 - phi(SoC)) * R_total  (fixed split, Ablation A)",
                          "OCV, tau": "frozen at BOL",
                          "SoC(t)": "SoC0 + integral(I_known)/Q(k); Q(k) from a fade curve, not from here",
                          "valid ranges": "OCV outside soc_valid and R_total outside the two-rate overlap are extrapolated"},
                   provenance=dict(csv=os.path.abspath(args.csv), impedance=args.impedance, bol_rpt=args.bol_rpt,
                                   fit_events=[ev["k"] for ev in fit_ev], val_events=[ev["k"] for ev in val_ev]))
    with open(os.path.join(args.out, f"{name}_ecm_bol.json"), "w") as fh:
        json.dump(to_jsonable(handoff), fh, indent=1)

    settings = [k for k, v in list(globals().items())
                if k.isupper() and isinstance(v, (int, float, str, bool, type(None), tuple))]
    summary = dict(cell=name, runtime_s=round(time.time() - t_start, 1), q_bol_Ah=q_bol, q_source=blk["q_src"],
                   n_pulses=len(fits), n_pulses_used=len(good), tau_used_s=tau_used, resistance=rinfo,
                   phi_at_pulse_socs={f"{f['soc']:.2f}": float(np.interp(f["soc"], grid, model.phi)) for f in good},
                   tierA=tierA, tierB=tierB, joint_refine=refine_info,
                   tierC=ocv_df[ocv_df["rpt"] == args.bol_rpt][["soc", "kind", "err_mV"]].to_dict("records") if len(ocv_df) else [],
                   settings={k: globals()[k] for k in settings})
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(to_jsonable(summary), fh, indent=1)

    if MAKE_PLOTS:
        plot_pulses(cell, fits, os.path.join(args.out, f"{name}_pulses.png"))
        plot_ecm(model, ocv_info, tr, good, ocv_df, imp_row, boot_r, ocv_dis, os.path.join(args.out, f"{name}_ecm.png"))
        plot_validation(cell, model, val_ev, q_bol, pooled_b, half_res, os.path.join(args.out, f"{name}_validation.png"))
        plot_aging(floor_df, os.path.join(args.out, f"{name}_aging_floor.png"))
        plot_rpt_series(ocv_df, os.path.join(args.out, f"{name}_rpt_steps.png"))
    print(f"Done in {time.time() - t_start:.1f} s. Outputs in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())