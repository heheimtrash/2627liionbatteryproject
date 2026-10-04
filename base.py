# this code is an accurate recreation of Ahuja et al. 2026's methodology, article "Lithium-ion battery State of Health
# estimation based solely on temperature sensing", J. Power Sources 666 239068
# dataset is CMU eVTOL dataset, by Bills et al.: https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830
# required extra packages: casadi, do-mpc, matplotlib, pandas

import numpy as np
from scipy import curve_fit
import sys
import json # needed as a summary.json file is produced 
from casadi import *
import os
import time
from pathlib import Path # need this to be able to read files from different path
from dataclasses import dataclass, asdict, field

import do_mpc 
import matplotlib.pyplot as plt
import pandas as pd

# my understanding: alpha is heat dissipation, leads to beta which is surface temp, leads to soh
# questioning if dataclass should be used? it's slightly more convenient in this situation
# variables explicitly stated by the paper
N_OF_HORIZON = 30
T_RISE_MAX = 60.0
RF_OVER_R0 = 2.0
N_ALPHA_EVENTS = 5 # first 5 cycles
ALPHA_PAPER = 2.2e-3 # thermal dissipation constant
ARRIVAL_WEIGHT = 1e-6 # eq 9 on paper, no arrival cost
N_BETA0_EVENTS = 5
TAIL_FRAC = 0.25
SMOOTH_HALF_WINDOW = 25 # +- 25 cycle moving average for graphs
T_AMB_MODE = "exp_fit"
ALPHA_MODE = "ls"

# assumed/ambiguous variables required for mhe in this scenario
CC_CURRENT_A = 3.0
CC_CURRENT_TOL_A = 0.3
REST_CURRENT_TOL_A = 0.05 
MIN_REST_S = 300.0 #shorter rest phases are ignored
MIN_CC_SAMPLES = 20 #shorter cc phases are ignored
EXCLUDE_AFTER_RPT = True #starts from 0% after the reference performance test cycle
LN_EPS_K = 0.3 # needed to drop rest samples with difference in battery temp vs ambient temp, before taking log() in eq 4 of paper
BETA_BOOTSTRAP_UB = 0.1
SIGMA_T_K = None
Q_T_K = 0.02 # purpose of line below but std on T per step (K)
Q_BETA_KPS = 2e-4 # process noise on standard deviation (std) on beta each step (K/s)
STRIDE = 1
T_AMB_FIXED = 23.0

SETTING_NAMES = [k for k in list (globals()) if k.isupper]

# load data
columns_from_dataset = ["time_s", "Ecell_V", "I_mA", "Temperature__C", "Ns"] # pulled from dataset csv file

def load_cell(path: str | Path) -> pd.DataFrame: 
    df = pd.read_csv(path, usecols=columns_from_dataset)
    df = df.sort_values("time_s") ###
    df["I_A"] = df["I_mA"] / 1000.0 #conversion from mA to A for current measurements, 1000.0 to keep it as float type
    return df

def funcruns(values: np.ndarray) -> list[tuple[int, int]]: # index ranges of consecutive equal values
    edges = np.r_[0, np.flatnonzero(np.diff(values) != 0) +1, len(values) ]
    return list(zip(edges[:-1], edges[1:]))

def extractevents(df: pd.DataFrame) -> list[dict]:
    t = df["time_s"].to_numpy()
    T = df["Temperature__C"].to_numpy()
    V = df["Ecell_V"].to_numpy()
    I = df["I_A"].to_numpy()
    ns = df["Ns"].to_numpy()
    events, rpt_times = [], []
    runs = funcruns(ns)
    after_rpt = False
    for j, (a, b) in enumerate(runs):
        I_med = np.median(I[a:b])
        if ns[a] == 0 and 0.3 < I_med < 0.7 * CC_CURRENT_A:
            rpt_times.append(t[a])
            after_rpt = True
        if abs(I_med - CC_CURRENT_A) > CC_CURRENT_TOL_A or (b-a) < MIN_CC_SAMPLES:
            continue
        skip, after_rpt = after_rpt and EXCLUDE_AFTER_RPT, False
        if skip:
            continue

        ev = dict(event=len(events), t0=t[a],
                  cc_t = t[a:b] - t[a], cc_T=T[a:b].copy(), rest = None,
                  V_cc0 = V[a], I_cc0 = I[a], V_rest_end = np.nan, I_rest_end = np.nan)

        if j > 0:
            ra, rb = funcruns[j-1]
            is_rest = np.all(np.abs(I[ra:rb]) < REST_CURRENT_TOL_A)
            duration = t[rb-1] - t[ra]
            if is_rest and duration >= MIN_REST_S:
                ev["rest"] = dict(t=t[ra:rb] - t[ra], T = T[ra:rb].copy())
                ev["V_rest_end"], ev["I_rest_end"] = V[rb-1], I[rb-1]
        events.append(ev)
    return events, np.asarray(rpt_times)

def resample_uniform(t: np.ndarray, y: np.ndarray, dt: float) -> np.ndarray: # linear interpolation to t=0, dt, 2dt and so on
    grid = np.arange(0.0, t[-1] + 1e-9, dt)
    return np.interp(grid, t, y)

#amb temperature and alpha, ahuja et al. section 2.1 up to equation 5
def funcexp_fit(t: np.ndarray, T: np.ndarray): # 3 parameter fit of T = T_inf + dTinitial * e^(-a * t), returns list of (T_inf, a, rmse)
    f = lambda x, Ta, dT0, a: Ta + dT0 * np.exp(-a * x)  # noqa: E731
    p0 = [T[-1] - 1.0, T[0] - T[-1] + 1.0, 2e-3]
    p, _ = curve_fit(f, t, T, p0=p0, maxfev=20000)
    return p[0], p[2], float(np.sqrt(np.mean((f(t, *p) - T) ** 2)))

def infer_t_amb(rest: dict | None, fallback: float, t_amb_bol: float = np.nan) -> float: # from my understanding,
    # T_amb is inferred from constant state minimum surface temperature during rest phases instead of reported amb (4.1-5)
    if T_AMB_MODE == "fixed":
        return T_AMB_FIXED
    if T_AMB_MODE == "exp_fit_bol" and np.isfinite(t_amb_bol):
        return t_amb_bol
    if rest is None:
        return fallback
    t, T = rest["t"], rest["T"]
    if T_AMB_MODE == "rest_min":
        tail = T[t >= t[-1] - 120.0]
        return float(np.mean(np.sort(tail)[:max(1, len(tail) // 4)]))
    if T_AMB_MODE in ("exp_fit", "expt_fit_bol"):
        return float(funcexp_fit(t, T)[0])
    raise ValueError(T_AMB_MODE)

def estimate_alpha_ls(t: np.ndarray, T: np.ndarray, t_amb: float, ln_eps: float) -> float: # paper equation 4 and 5
    m = (T - t_amb) > ln_eps
    if m.sum() < 5:
        return np.nan
    tau = t[m] - t[m][0]
    Z = -(np.log(T[m] - t_amb) - np.log(T[m][0] - t_amb))
    Phi = tau[:, None]
    return float(np.linalg.lstsq(Phi, Z, rcond=None)[0][0])

def calibrate_alpha(events_per_cell: list[list[dict]]) -> dict: # alpha from first 5 rest phases averaged, returns std of temp sensor (sigma_T) from rest phase fit
    alphas, resid, t_amb_bol = [], [], []
    for events in events_per_cell:
        rests = [e for e in events if e["rest"] is not None][: N_ALPHA_EVENTS]
        t_amb_bol.append(float(np.mean([funcexp_fit(e["rest"]["t"], e["rest"]["T"][0]) for e in rests])))
        for e in rests:
            t, T = e["rest"]["t"], e["rest"]["T"]
            t_amb = infer_t_amb(e["rest"], fallback=np.nan, t_amb_bol=t_amb_bol[-1])
            alphas.append(estimate_alpha_ls(t, T, t_amb, LN_EPS_K))
            k=5
            if len(T) > 3 * k:
                sm = np.convolve(T, np.ones(k) / k, mode="valid")
                resid.append(T[k // 2: k // 2 + len(sm)] - sm)
    alpha_ls = float(np.nanmean(alphas))
    sigma_T = float(np.std(np.concatenate(resid))) if resid else 0.05
    alpha = ALPHA_PAPER if ALPHA_MODE == "paper" else alpha_ls
    return dict(alpha=alpha, alpha_ls=alpha_ls, alpha_each=alphas, sigma_T=max(sigma_T, 0.01),
                t_amb_bol=t_amb_bol)

# mhe
# mhe for x=[T, beta] with dT/dt = alpha (T_amb - T) + beta where d beta/dt =0
# Eq 9 penalizes discrete model residual, so do-mpc is set as discrete
# T = e^{-a dt} T + (1-e^{-a dt}) T_amb + (1-e^{-a dt})/a * beta + w_T, process noise (w)
# has units K and K/s each step, making it match equation 9

def __init__(self, alpha: float, dt: float, beta_ub: float, sigma_T: float):
    self.dt = dt
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
    mhe.settings.n_horizon = N_HORIZON
    mhe.settings.t_step = dt
    mhe.settings.meas_from_data = True
    mhe.settings.store_full_solution = False
    mhe.settings.store_lagr_multiplier = False
    mhe.settings.nlpsol_opts = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0}

    # Eq. 9: ||y - C x||^2_{R_MHE} + ||x_{j+1} - (A x_j + B u_j)||^2_{Q_MHE}
    # do-mpc: P_v weights v (measurement residual), P_w weights w (model residual),
    # P_x weights the arrival cost, which the paper does not have -> negligible weight.
    P_v = np.array([[1.0 / sigma_T ** 2]])
    P_w = np.diag([1.0 / Q_T_K ** 2, 1.0 / Q_BETA_KPS ** 2])
    P_x = ARRIVAL_WEIGHT * np.diag([1.0 / sigma_T ** 2, 1.0 / Q_BETA_KPS ** 2])
    mhe.set_default_objective(P_x=P_x, P_v=P_v, P_w=P_w)

    # Eq. 10 bounds: 0 <= beta <= beta_f ;  T_amb <= T <= T_amb + 60 (depends on T_amb)
    mhe.bounds["lower", "_x", "beta"] = 0.0
    mhe.bounds["upper", "_x", "beta"] = beta_ub
    mhe.set_nl_cons("T_lower", T_amb - T, ub=0.0)
    mhe.set_nl_cons("T_upper", T - T_amb - T_RISE_MAX_K, ub=0.0)

    self._t_amb = 0.0
    tvp = mhe.get_tvp_template()

    def tvp_fun(_t_now):
        tvp["_tvp", :, "T_amb"] = self._t_amb
        return tvp

    mhe.set_tvp_fun(tvp_fun)
    mhe.setup()
    self.mhe = mhe

def run_event(self, T_cc: np.ndarray, t_amb: float) -> tuple[np.ndarray, np.ndarray]:
    """Feed one CC event's uniformly sampled T through the MHE.
    Returns (T_hat, beta_hat), one value per sample."""
    mhe = self.mhe
    self._t_amb = float(t_amb)
    mhe.reset_history()
    # [ASSUMED] the paper starts from "zero initial conditions"; T_hat = 0 would violate
    # T_hat >= T_amb, so start at the first reading and a small beta.
    x0 = np.array([[max(T_cc[0], t_amb)], [BETA_INIT_KPS]])
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
