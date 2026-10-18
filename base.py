# this code is an accurate recreation of Ahuja et al. 2026's methodology, article "Lithium-ion battery State of Health
# estimation based solely on temperature sensing", J. Power Sources 666 239068
# dataset is CMU eVTOL dataset, by Bills et al.: https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830
# required extra packages: casadi, do-mpc[full], matplotlib, pandas

import numpy as np
import argparse
from scipy.optimize import curve_fit
import sys
import json # needed as a summary.json file is produced 
import casadi as ca
import os
import time
from pathlib import Path # need this to be able to read files from different path
from dataclasses import dataclass, asdict, field

import do_mpc 
import matplotlib.pyplot as plt
import pandas as pd

# my understanding: alpha is heat dissipation, leads to beta which is surface temp, leads to soh
# questioning if dataclass should be used? it's slightly more convenient in this situation
import argparse
import json
import time
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*feature.*not available.*")
warnings.filterwarnings("ignore", message=".*approximateMPC.*")
import casadi as _ca  # noqa: E402
try:
    _ca.GlobalOptions.setNumpyMode(-1)   # silence casadi numpy-mode notice
except AttributeError:
    pass
import do_mpc  # noqa: E402


@dataclass
class Config:
    # ---- event detection (dataset specific) -------------------------------------------
    cc_current_A: float = 3.0          # nominal CC charge current to select (1C for VTC-6 = 3 A)
    cc_current_tol_A: float = 0.3      # |median I - cc_current_A| tolerance
    rest_current_tol_A: float = 0.05   # |I| below this counts as rest
    min_rest_s: float = 300.0          # [ASSUMED] shorter rests are not used for alpha / T_amb
    min_cc_samples: int = 20           # [ASSUMED] skip CC events shorter than this
    exclude_after_rpt: bool = True     # [ASSUMED] drop the first 1C charge after each RPT: it starts
                                       # from 0 % SoC after a long 21 C rest (not a mission rest),
                                       # so neither its T(t) nor its IR drop is comparable

    # ---- thermal model / alpha ---------------------------------------------------------
    n_alpha_events: int = 5            # [PAPER] alpha from the first five pre-charge rests
    alpha_mode: str = "ls"             # "ls": fit from data (paper Eqs. 3-5) | "paper": use 0.0022
    alpha_paper: float = 2.2e-3        # [PAPER] 1/s, average over all of their tests
    t_amb_mode: str = "exp_fit"        # "rest_min":    [PAPER, literal] minimum surface T in the rest
                                       #                before each charge
                                       # "exp_fit_bol": [ASSUMED] asymptote of a 3-parameter exponential
                                       #                fit, averaged over the first five rests and held
                                       #                constant (chamber T is constant)
                                       # "exp_fit":     same fit, but per event
                                       # "fixed":       t_amb_fixed for every event
    t_amb_fixed: float = 23.0          # only for t_amb_mode == "fixed"
    ln_eps_K: float = 0.3              # [ASSUMED] drop rest samples with T - T_amb < this before
                                       # taking log() in Eq. 4 (log(0) at the end of the rest)

    # ---- MHE -------------------------------------------------------------------------------
    dt_s: float | None = None          # sample time; None -> median of the CC sampling (30 s here)
    n_horizon: int = 30                # [PAPER] horizon of 30 samples
    t_rise_max_K: float = 60.0         # [PAPER] T_amb <= T_hat <= T_amb + 60
    rf_over_r0: float = 2.0            # [PAPER] EOL resistance = 2 x initial -> beta_f = 2 beta_0
    beta_bootstrap_ub: float = 0.1     # [ASSUMED] K/s, loose bound for the first-5-event pass
    sigma_T_K: float | None = None     # meas. noise std; None -> estimated from rest-phase residuals
    q_T_K: float = 0.02                # [ASSUMED] process-noise std on T per step  [K]
    q_beta_Kps: float = 2e-4           # [ASSUMED] process-noise std on beta per step [K/s]
    arrival_weight: float = 1e-6       # [PAPER] Eq. 9 has no arrival cost -> make it negligible
    beta_init_Kps: float = 1e-3        # [ASSUMED] initial guess for beta at the start of each CC

    # ---- SoH_p -------------------------------------------------------------------------------
    n_beta0_events: int = 5            # [PAPER] beta_0 = mean of beta_k over first five cycles
    tail_frac: float = 0.25            # [PAPER] beta_k = mean of beta_hat over last quarter of CC
    smooth_half_window: int = 25       # [PAPER] centred +/- 25-cycle moving average for plots/RMSE

    # ---- run control -----------------------------------------------------------------------
    stride: int = 1                    # process every n-th CC event (for quick runs)
    max_events: int | None = None


# Data loading and event extraction
USECOLS = ["time_s", "Ecell_V", "I_mA", "Temperature__C", "Ns"]


def load_cell(path: str | Path) -> pd.DataFrame:
# Load one Bills et al. cell CSV, sorted chronologically. Current converted to A
    df = pd.read_csv(path, usecols=USECOLS)
    df = df.sort_values("time_s", kind="stable").reset_index(drop=True)
    df["I_A"] = df["I_mA"] / 1000.0
    return df


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
# Index ranges [a, b) of consecutive equal values
    edges = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1, len(values)]
    return list(zip(edges[:-1], edges[1:]))


def extract_events(df: pd.DataFrame, cfg: Config) -> list[dict]:
# one dict per cc charge event
    t = df["time_s"].to_numpy()
    T = df["Temperature__C"].to_numpy()
    V = df["Ecell_V"].to_numpy()
    I = df["I_A"].to_numpy()
    ns = df["Ns"].to_numpy()

    events, rpt_times = [], []
    runs = _runs(ns)
    after_rpt = False
    for j, (a, b) in enumerate(runs):
        I_med = np.median(I[a:b])
        if ns[a] == 0 and 0.3 < I_med < 0.7 * cfg.cc_current_A:
            rpt_times.append(t[a])  # slow (C/2) RPT charge
            after_rpt = True
        if abs(I_med - cfg.cc_current_A) > cfg.cc_current_tol_A or (b - a) < cfg.min_cc_samples:
            continue
        skip, after_rpt = after_rpt and cfg.exclude_after_rpt, False
        if skip:
            continue

        ev = dict(event=len(events), t0=t[a],
                  cc_t=t[a:b] - t[a], cc_T=T[a:b].copy(), rest=None,
                  # [DEV-ONLY] quantities for the IR-drop reference
                  V_cc0=V[a], I_cc0=I[a], V_rest_end=np.nan, I_rest_end=np.nan)

        if j > 0:
            ra, rb = runs[j - 1]
            is_rest = np.all(np.abs(I[ra:rb]) < cfg.rest_current_tol_A)
            dur = t[rb - 1] - t[ra]
            if is_rest and dur >= cfg.min_rest_s:
                ev["rest"] = dict(t=t[ra:rb] - t[ra], T=T[ra:rb].copy())
                ev["V_rest_end"], ev["I_rest_end"] = V[rb - 1], I[rb - 1]
        events.append(ev)
    return events, np.asarray(rpt_times)


def resample_uniform(t: np.ndarray, y: np.ndarray, dt: float) -> np.ndarray:
# linear interpolation
    grid = np.arange(0.0, t[-1] + 1e-9, dt)
    return np.interp(grid, t, y)



# Ambient temperature and alpha (paper Sec. 2.1, Eqs. 2-5)

def _exp_fit(t: np.ndarray, T: np.ndarray):
    from scipy.optimize import curve_fit
    f = lambda x, Ta, dT0, a: Ta + dT0 * np.exp(-a * x)  # noqa: E731
    p0 = [T[-1] - 1.0, T[0] - T[-1] + 1.0, 2e-3]
    p, _ = curve_fit(f, t, T, p0=p0, maxfev=20000)
    return p[0], p[2], float(np.sqrt(np.mean((f(t, *p) - T) ** 2)))


def infer_t_amb(rest: dict | None, cfg: Config, fallback: float, t_amb_bol: float = np.nan) -> float:
#    T_amb for one event from its pre-charge rest.
#
#    [PAPER] "T_amb was inferred from the steady-state minimum surface temperature during rest
#    phases, rather than the reported ambient" (Sec. 4.1 and Sec. 5). take the minimum over
#    the last ~2 min of the rest (= the minimum, since T decays monotonically, but robust to a
#    single noisy low sample).
#    NOTE: in this dataset the rest ends on a temperature threshold (~27.4 C), so this value is
#    above the true asymptote. "exp_fit" is offered as an alternative.
#    
    if cfg.t_amb_mode == "fixed":
        return cfg.t_amb_fixed
    if cfg.t_amb_mode == "exp_fit_bol" and np.isfinite(t_amb_bol):
        return t_amb_bol
    if rest is None:
        return fallback
    t, T = rest["t"], rest["T"]
    if cfg.t_amb_mode == "rest_min":
        tail = T[t >= t[-1] - 120.0]
        return float(np.mean(np.sort(tail)[: max(1, len(tail) // 4)]))
    if cfg.t_amb_mode in ("exp_fit", "exp_fit_bol"):
        return float(_exp_fit(t, T)[0])
    raise ValueError(cfg.t_amb_mode)


def estimate_alpha_ls(t: np.ndarray, T: np.ndarray, t_amb: float, ln_eps: float) -> float:
#    Paper Eqs. 4-5:  Z = -[ln(T - T_amb) - ln(T0 - T_amb)] = alpha * t,
#    zero-intercept batch least squares alpha = (Phi' Phi)^-1 Phi' Z with Phi = t.
    m = (T - t_amb) > ln_eps
    if m.sum() < 5:
        return np.nan
    tau = t[m] - t[m][0]
    Z = -(np.log(T[m] - t_amb) - np.log(T[m][0] - t_amb))
    Phi = tau[:, None]
    return float(np.linalg.lstsq(Phi, Z, rcond=None)[0][0])


def calibrate_alpha(events_per_cell: list[list[dict]], cfg: Config) -> dict:
#    [PAPER] alpha from the first five rests of each test, averaged over tests.
#    Also returns sigma_T estimated from the rest-phase fit residuals (paper: R tuned on
#    the variance of the surface temperature sensor).
    alphas, resid, t_amb_bol = [], [], []
    for events in events_per_cell:
        rests = [e for e in events if e["rest"] is not None][: cfg.n_alpha_events]
        # cell-level ambient from the first five rests (used by t_amb_mode == "exp_fit_bol")
        t_amb_bol.append(float(np.mean([_exp_fit(e["rest"]["t"], e["rest"]["T"])[0] for e in rests])))
        for e in rests:
            t, T = e["rest"]["t"], e["rest"]["T"]
            t_amb = infer_t_amb(e["rest"], cfg, fallback=np.nan, t_amb_bol=t_amb_bol[-1])
            alphas.append(estimate_alpha_ls(t, T, t_amb, cfg.ln_eps_K))
            # high-pass residual for sensor noise: deviation from a 5-sample moving average
            k = 5
            if len(T) > 3 * k:
                sm = np.convolve(T, np.ones(k) / k, mode="valid")
                resid.append(T[k // 2: k // 2 + len(sm)] - sm)
    alpha_ls = float(np.nanmean(alphas))
    sigma_T = float(np.std(np.concatenate(resid))) if resid else 0.05
    alpha = cfg.alpha_paper if cfg.alpha_mode == "paper" else alpha_ls
    return dict(alpha=alpha, alpha_ls=alpha_ls, alpha_each=alphas, sigma_T=max(sigma_T, 0.01),
                t_amb_bol=t_amb_bol)



# do-mpc Moving Horizon Estimator (paper Eqs. 6, 9, 10)

class ThermalMHE:
#    MHE for x = [T, beta] with the paper's model Tdot = alpha (T_amb - T) + beta, betadot = 0.
#
#    The paper's Eq. 9 penalises the DISCRETE model residual x_{j+1} - (A x_j + B u_j), so the
#    model is built as a do-mpc *discrete* model using the exact zero-order-hold discretisation:
#        T+    = e^{-a dt} T + (1 - e^{-a dt}) T_amb + (1 - e^{-a dt})/a * beta + w_T
#        beta+ = beta + w_beta
#    Process noise w therefore has units [K] and [K/s] per step, matching Q_MHE in Eq. 9.
#    T_amb enters as a time-varying parameter (constant over one CC event), so one compiled
#    MHE is reused for all events.
#    

    def __init__(self, alpha: float, dt: float, beta_ub: float, sigma_T: float, cfg: Config):
        self.dt, self.cfg = dt, cfg
        model = do_mpc.model.Model("discrete")
        T = model.set_variable("_x", "T")
        beta = model.set_variable("_x", "beta")
        T_amb = model.set_variable("_tvp", "T_amb")

        ead = float(np.exp(-alpha * dt))
        model.set_rhs("T", ead * T + (1.0 - ead) * T_amb + (1.0 - ead) / alpha * beta,
                      process_noise=True)
        model.set_rhs("beta", beta, process_noise=True)       # beta constant during CC
        model.set_meas("T_meas", T, meas_noise=True)            # y = C x + v, C = [1 0]
        model.setup()

        mhe = do_mpc.estimator.MHE(model)
        mhe.settings.n_horizon = cfg.n_horizon
        mhe.settings.t_step = dt
        mhe.settings.meas_from_data = True
        mhe.settings.store_full_solution = False
        mhe.settings.store_lagr_multiplier = False
        mhe.settings.nlpsol_opts = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0}

        # Eq. 9: ||y - C x||^2_{R_MHE} + ||x_{j+1} - (A x_j + B u_j)||^2_{Q_MHE}
        # do-mpc: P_v weights v (measurement residual), P_w weights w (model residual),
        # P_x weights the arrival cost, which the paper does not have -> negligible weight.
        P_v = np.array([[1.0 / sigma_T ** 2]])
        P_w = np.diag([1.0 / cfg.q_T_K ** 2, 1.0 / cfg.q_beta_Kps ** 2])
        P_x = cfg.arrival_weight * np.diag([1.0 / sigma_T ** 2, 1.0 / cfg.q_beta_Kps ** 2])
        mhe.set_default_objective(P_x=P_x, P_v=P_v, P_w=P_w)

        # Eq. 10 bounds: 0 <= beta <= beta_f ;  T_amb <= T <= T_amb + 60 (depends on T_amb)
        mhe.bounds["lower", "_x", "beta"] = 0.0
        mhe.bounds["upper", "_x", "beta"] = beta_ub
        mhe.set_nl_cons("T_lower", T_amb - T, ub=0.0)
        mhe.set_nl_cons("T_upper", T - T_amb - cfg.t_rise_max_K, ub=0.0)

        self._t_amb = 0.0
        tvp = mhe.get_tvp_template()

        def tvp_fun(_t_now):
            tvp["_tvp", :, "T_amb"] = self._t_amb
            return tvp

        mhe.set_tvp_fun(tvp_fun)
        mhe.setup()
        self.mhe = mhe

    def run_event(self, T_cc: np.ndarray, t_amb: float) -> tuple[np.ndarray, np.ndarray]:
# Feed one CC event's uniformly sampled T through the MHE.
# Returns (T_hat, beta_hat), one value per sample
        mhe = self.mhe
        self._t_amb = float(t_amb)
        mhe.reset_history()
        # [ASSUMED] the paper starts from "zero initial conditions"; T_hat = 0 would violate
        # T_hat >= T_amb, so start at the first reading and a small beta.
        x0 = np.array([[max(T_cc[0], t_amb)], [self.cfg.beta_init_Kps]])
        mhe.x0 = x0
        mhe.set_initial_guess()
        # do-mpc warm-starts from the previous event's solution; re-seed every node instead
        mhe.opt_x_num["_x", :, :] = x0 / mhe._x_scaling.cat.full()
        mhe.opt_x_num["_w", :] = 0.0
        mhe.opt_x_num["_v", :] = 0.0

        T_hat = np.empty(len(T_cc))
        beta_hat = np.empty(len(T_cc))
        for k, y in enumerate(T_cc):
            x = mhe.make_step(np.array([[y]]))
            T_hat[k], beta_hat[k] = float(x[0, 0]), float(x[1, 0])
        return T_hat, beta_hat


# beta_k, SoH_p, references

def beta_k_from_trace(beta_hat: np.ndarray, tail_frac: float) -> float:
#    [PAPER] steady-state beta for one CC event = mean over the last quarter of the CC phase
    n0 = int(np.floor(len(beta_hat) * (1.0 - tail_frac)))
    return float(np.mean(beta_hat[n0:]))


def soh_from_metric(x: np.ndarray, n0: int, inverse: bool = True) -> np.ndarray:
#    SoH_p = x_0 / x_k * 100 (Eqs. 12-13), x_0 = mean over the first n0 finite values
    x = np.asarray(x, float)
    x0 = np.nanmean(x[np.isfinite(x)][:n0])
    return 100.0 * (x0 / x if inverse else x / x0)


def centred_moving_average(x: np.ndarray, half: int) -> np.ndarray:
    s = pd.Series(np.asarray(x, float))
    return s.rolling(2 * half + 1, center=True, min_periods=1).mean().to_numpy()


def ir_drop_resistance(ev: dict) -> float:
#    [DEV-ONLY] Paper Eq. 13 reference: R = dV/dI between the last rest sample and the
#    first CC sample. With 30 s logging this is a ~0-30 s pulse resistance (R0 + part of Rp),
#    not a true 'instantaneous' ohmic resistance.
    dI = ev["I_cc0"] - ev["I_rest_end"]
    if not np.isfinite(dI) or abs(dI) < 0.5:
        return np.nan
    return float(abs((ev["V_cc0"] - ev["V_rest_end"]) / dI))


def load_impedance(path: str | Path, rpt_times: np.ndarray, events: list[dict]) -> pd.DataFrame:
#    [DEV-ONLY] Dataset's pulse-resistance file (R after 1 s and 30 s at 20 % / 60 % SoC,
#    measured during the RPTs). Row i is aligned to the i-th RPT in the time series and mapped
#    to the chronological index of the first 1C CC event after it
    imp = pd.read_csv(path)
    ev_t0 = np.array([e["t0"] for e in events])
    idx = []
    for i in range(len(imp)):
        if i < len(rpt_times):
            later = np.flatnonzero(ev_t0 > rpt_times[i])
            idx.append(int(later[0]) if len(later) else np.nan)
        else:
            idx.append(np.nan)
    imp["event"] = idx
    return imp


def rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))



# Pipeline

def run_mhe_over_events(est: ThermalMHE, events: list[dict], cfg: Config, keep_traces=(),
                        t_amb_bol: float = np.nan):
# Run the MHE over a list of events; returns per-event table and selected traces
    rows, traces = [], {}
    t_amb_last = np.nan
    t_start = time.time()
    for n, ev in enumerate(events):
        t_amb = infer_t_amb(ev["rest"], cfg, fallback=t_amb_last, t_amb_bol=t_amb_bol)
        if not np.isfinite(t_amb):
            t_amb = float(ev["cc_T"][0])
        t_amb_last = t_amb
        T_cc = resample_uniform(ev["cc_t"], ev["cc_T"], est.dt)
        T_hat, b_hat = est.run_event(T_cc, t_amb)
        rows.append(dict(event=ev["event"], t0_s=ev["t0"], T_amb=t_amb, cc_samples=len(T_cc),
                         cc_dur_s=float(ev["cc_t"][-1]), T_rise_K=float(T_cc.max() - T_cc[0]),
                         beta_k=beta_k_from_trace(b_hat, cfg.tail_frac),
                         fit_rmse_K=float(np.sqrt(np.mean((T_hat - T_cc) ** 2))),
                         R_ir=ir_drop_resistance(ev), has_rest=ev["rest"] is not None))
        if ev["event"] in keep_traces:
            traces[ev["event"]] = dict(t=np.arange(len(T_cc)) * est.dt, T=T_cc, T_hat=T_hat,
                                       beta=b_hat, T_amb=t_amb)
        if (n + 1) % 50 == 0:
            el = time.time() - t_start
            print(f"    {n + 1}/{len(events)} events  ({el:.0f} s, "
                  f"{1e3 * el / sum(r['cc_samples'] for r in rows):.1f} ms/step)", flush=True)
    return pd.DataFrame(rows), traces


def process_cell(name: str, events: list[dict], alpha_info: dict, cfg: Config, dt: float,
                 t_amb_bol: float = np.nan):
    alpha, sigma_T = alpha_info["alpha"], cfg.sigma_T_K or alpha_info["sigma_T"]
    sel = events[:: cfg.stride]
    if cfg.max_events:
        sel = sel[: cfg.max_events]
    # always include the first n_beta0_events so beta_0 is defined on the true BOL cycles
    head = events[: cfg.n_beta0_events]
    sel = head + [e for e in sel if e["event"] >= cfg.n_beta0_events]

    # Pass A: first five events with a loose beta bound -> beta_0 (no cell internals needed)
    print(f"  [{name}] pass A: beta_0 from first {cfg.n_beta0_events} events")
    est_A = ThermalMHE(alpha, dt, cfg.beta_bootstrap_ub, sigma_T, cfg)
    tab_A, _ = run_mhe_over_events(est_A, head, cfg, t_amb_bol=t_amb_bol)
    beta_0 = float(tab_A["beta_k"].mean())
    beta_f = cfg.rf_over_r0 * beta_0       # [PAPER] Eq. 11 with R_f = 2 R_0
    print(f"  [{name}] beta_0 = {beta_0:.3e} K/s  ->  beta_f = {beta_f:.3e} K/s")

    # Pass B: all selected events with the paper's bound
    keep = (sel[0]["event"], sel[-1]["event"])
    print(f"  [{name}] pass B: {len(sel)} events, horizon {cfg.n_horizon} x {dt:.0f} s")
    est_B = ThermalMHE(alpha, dt, beta_f, sigma_T, cfg)
    tab, traces = run_mhe_over_events(est_B, sel, cfg, keep_traces=keep, t_amb_bol=t_amb_bol)

    n0 = cfg.n_beta0_events
    h = cfg.smooth_half_window
    # paper smooths over +/-25 *cycles*; with stride s that is +/- 25/s processed events
    h_ev = max(1, int(round(h / cfg.stride)))
    tab["beta_k_smooth"] = centred_moving_average(tab["beta_k"], h_ev)
    tab["R_ir_smooth"] = centred_moving_average(tab["R_ir"], h_ev)
    tab["SoH_T"] = soh_from_metric(tab["beta_k"], n0)                  # Eq. 12, raw
    tab["SoH_T_smooth"] = soh_from_metric(tab["beta_k_smooth"], n0)    # Eq. 12, smoothed beta
    tab["SoH_IR"] = soh_from_metric(tab["R_ir"], n0)                    # Eq. 13, raw  [DEV-ONLY]
    tab["SoH_IR_smooth"] = soh_from_metric(tab["R_ir_smooth"], n0)
    tab["R_ratio_T"] = tab["beta_k"] / beta_0       # = R_k / R_0, handoff quantity to Step 2
    return tab, traces, dict(beta_0=beta_0, beta_f=beta_f, sigma_T=sigma_T)


def summarise(tab: pd.DataFrame, cfg: Config) -> dict:
    last = tab.iloc[-1]
    return dict(
        events=int(len(tab)),
        rmse_raw=rmse(tab["SoH_T"], tab["SoH_IR"]),
        rmse_smoothed=rmse(tab["SoH_T_smooth"], tab["SoH_IR_smooth"]),
        sigma_bar_T=float(np.nanmean(pd.Series(tab["SoH_T"]).rolling(
            2 * max(1, cfg.smooth_half_window // cfg.stride) + 1, center=True, min_periods=3).std())),
        sigma_bar_IR=float(np.nanmean(pd.Series(tab["SoH_IR"]).rolling(
            2 * max(1, cfg.smooth_half_window // cfg.stride) + 1, center=True, min_periods=3).std())),
        eol_SoH_T_smooth=float(last["SoH_T_smooth"]),
        eol_SoH_IR_smooth=float(last["SoH_IR_smooth"]),
        median_fit_rmse_K=float(tab["fit_rmse_K"].median()),
    )



# Plots

C_IR, C_MHE, C_IMP, C_MEAS = "#3d3d3d", "#2a6fdb", "#d9730d", "#8a8a8a"


def _style(ax):
    ax.grid(True, color="#e6e6e6", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_alpha_fit(events, cfg, alpha, path, t_amb_bol=np.nan):
    import matplotlib.pyplot as plt
    e = next(e for e in events if e["rest"] is not None)
    t, T = e["rest"]["t"], e["rest"]["T"]
    t_amb = infer_t_amb(e["rest"], cfg, np.nan, t_amb_bol)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(t, T, ".", ms=3, color=C_MEAS, label="measured (first pre-charge rest)")
    ax.plot(t, t_amb + (T[0] - t_amb) * np.exp(-alpha * t), lw=2, color=C_MHE,
            label=f"Eq. 3, alpha = {alpha:.2e} 1/s")
    ax.axhline(t_amb, ls="--", lw=1, color=C_IR, label=f"T_amb = {t_amb:.2f} C ({cfg.t_amb_mode})")
    ax.set_xlabel("time in rest [s]"); ax.set_ylabel("surface T [C]")
    ax.legend(frameon=False, fontsize=8); _style(ax); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_traces(traces, path):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for (ev, tr), lab, ls in zip(sorted(traces.items()), ("BOL", "EOL"), ("-", "--")):
        a1.plot(tr["t"], tr["T"], ls, color=C_MEAS, lw=1.5, label=f"measured, {lab} (event {ev})")
        a1.plot(tr["t"], tr["T_hat"], ls, color=C_MHE, lw=1.5, label=f"MHE T_hat, {lab}")
        a2.plot(tr["t"], tr["beta"], ls, color=C_MHE, lw=1.8, label=f"{lab} (event {ev})")
        n0 = int(len(tr["beta"]) * 0.75)
        a2.axvspan(tr["t"][n0], tr["t"][-1], color=C_MHE, alpha=0.06)
    a1.set_xlabel("time in CC [s]"); a1.set_ylabel("surface T [C]"); a1.legend(frameon=False, fontsize=7)
    a2.set_xlabel("time in CC [s]"); a2.set_ylabel("beta_hat [K/s]")
    a2.set_title("shaded: last quarter used for beta_k", fontsize=8); a2.legend(frameon=False, fontsize=8)
    _style(a1); _style(a2); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_soh(tab, imp, path, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(tab["event"], tab["SoH_IR"], ".", ms=2, color=C_IR, alpha=0.25)
    ax.plot(tab["event"], tab["SoH_T"], ".", ms=2, color=C_MHE, alpha=0.25)
    ax.plot(tab["event"], tab["SoH_IR_smooth"], color=C_IR, lw=2, label="IR drop (Eq. 13), +/-25 avg")
    ax.plot(tab["event"], tab["SoH_T_smooth"], color=C_MHE, lw=2, label="MHE, temperature only (Eq. 12), +/-25 avg")
    if imp is not None and imp["event"].notna().any():
        m = imp["event"].notna()
        for col, mk in (("60%_30_second", "o"), ("60%_1_second", "s")):
            if col in imp:
                s = 100 * imp.loc[m, col].iloc[0] / imp.loc[m, col]
                ax.plot(imp.loc[m, "event"], s, mk, ms=6, mfc="white", mec=C_IMP, mew=1.5,
                        label=f"dataset pulse R, {col.replace('_', ' ')}")
    ax.set_xlabel("1C charge event (chronological)"); ax.set_ylabel("SoH_p [%]")
    ax.set_title(title, fontsize=10); ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)



# Main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", nargs="+", required=True, help="cell time-series CSV(s)")
    ap.add_argument("--impedance", nargs="*", default=[], help="matching *_impedance.csv (optional)")
    ap.add_argument("--out", default="results_step1")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--alpha-mode", choices=["ls", "paper"], default="ls")
    ap.add_argument("--t-amb-mode", choices=["rest_min", "exp_fit_bol", "exp_fit", "fixed"],
                    default="exp_fit")
    ap.add_argument("--t-amb-fixed", type=float, default=23.0)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--q-beta", type=float, default=Config.q_beta_Kps, help="beta process-noise std per step [K/s]")
    ap.add_argument("--q-T", type=float, default=Config.q_T_K, help="T process-noise std per step [K]")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)

    cfg = Config(stride=a.stride, max_events=a.max_events, alpha_mode=a.alpha_mode,
                 t_amb_mode=a.t_amb_mode, t_amb_fixed=a.t_amb_fixed, n_horizon=a.horizon,
                 q_beta_Kps=a.q_beta, q_T_K=a.q_T)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # 1) load and segment every cell
    cells = {}
    for p in a.csv:
        name = Path(p).stem
        print(f"loading {p} ...", flush=True)
        df = load_cell(p)
        events, rpt_times = extract_events(df, cfg)
        n_rest = sum(e["rest"] is not None for e in events)
        print(f"  {len(events)} CC events at ~{cfg.cc_current_A} A ({n_rest} with a usable rest), "
              f"{len(rpt_times)} RPT charges")
        cells[name] = dict(events=events, rpt_times=rpt_times)
        del df

    # 2) alpha (and sensor noise), averaged over all cells as in the paper
    ainfo = calibrate_alpha([c["events"] for c in cells.values()], cfg)
    print(f"T_amb(BOL exp fit) = {ainfo['t_amb_bol']}, alpha_LS = {ainfo['alpha_ls']:.3e} 1/s  (paper average {cfg.alpha_paper:.1e}); "
          f"using alpha = {ainfo['alpha']:.3e}; sigma_T = {ainfo['sigma_T']:.3f} K")

    # 3) sample time
    dts = [np.median(np.diff(e["cc_t"])) for c in cells.values() for e in c["events"]]
    dt = cfg.dt_s or float(np.round(np.median(dts), 3))
    print(f"MHE sample time = {dt:.1f} s, horizon = {cfg.n_horizon} samples ({cfg.n_horizon * dt:.0f} s)")

    summary = dict(config=asdict(cfg), alpha=ainfo["alpha"], alpha_ls=ainfo["alpha_ls"],
                   alpha_each=ainfo["alpha_each"], dt=dt, cells={})
    imp_paths = dict(zip([Path(p).stem for p in a.csv], a.impedance)) if a.impedance else {}

    for ci, (name, c) in enumerate(cells.items()):
        t0 = time.time()
        tab, traces, info = process_cell(name, c["events"], ainfo, cfg, dt, ainfo["t_amb_bol"][ci])
        info["t_amb_bol"] = ainfo["t_amb_bol"][ci]
        s = summarise(tab, cfg); s.update(info); s["runtime_s"] = time.time() - t0
        imp = None
        if name in imp_paths:
            imp = load_impedance(imp_paths[name], c["rpt_times"], c["events"])
            imp.to_csv(out / f"{name}_impedance_aligned.csv", index=False)
            # [DEV-ONLY] how well beta tracks the dataset's own pulse resistances
            m = imp["event"].notna()
            ev_idx = tab["event"].to_numpy()
            nearest = [int(np.argmin(np.abs(ev_idx - e))) for e in imp.loc[m, "event"]]
            est = tab["SoH_T_smooth"].to_numpy()[nearest]
            for col in [k for k in imp.columns if "second" in k]:
                ref = 100 * imp.loc[m, col].iloc[0] / imp.loc[m, col]
                s[f"rmse_vs_pulse_{col}"] = rmse(est, ref)
        summary["cells"][name] = s
        tab.to_csv(out / f"{name}_step1_events.csv", index=False)
        print(f"  [{name}] SoH_p RMSE vs IR drop: raw {s['rmse_raw']:.2f} %, smoothed {s['rmse_smoothed']:.2f} %"
              f"  | EOL SoH_p: T-based {s['eol_SoH_T_smooth']:.1f} %, IR {s['eol_SoH_IR_smooth']:.1f} %"
              f"  | {s['runtime_s']:.0f} s")
        if not a.no_plots:
            plot_traces(traces, out / f"{name}_beta_traces.png")
            plot_soh(tab, imp, out / f"{name}_soh.png",
                     f"{name}: SoH_p from surface temperature (do-mpc MHE) vs voltage-based references")
    if not a.no_plots:
        first = next(iter(cells.values()))["events"]
        plot_alpha_fit(first, cfg, ainfo["alpha"], out / "alpha_fit.png", ainfo["t_amb_bol"][0])

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"wrote results to {out.resolve()}")
    return summary


if __name__ == "__main__":
    main()
