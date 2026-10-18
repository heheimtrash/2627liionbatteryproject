# Step 3: extended MHE for x = [T, beta, SOC] with V(t) as derived output
# Deployment path:
#    pre-charge rest T(t) -> T_amb (Step 1, unchanged)
#    CC-charge T(t), T_amb -> T_hat, beta_hat (Step 1 thermal model, unchanged)
#    mission Ah (duty cycle), Q_fade(k) -> SoC0 prior = arrival cost on SoC (channel 1)
#    CC end time, charger V_CV -> V_hat(SoC_end) = V_CV, pseudo-measurement (channel 2, the "anchor")
#    [optional] beta follows R_BOL(SoC) -> SoC information from the shape of beta(t)  (channel 3, off by default)
#    SoC_hat(t), beta_k / beta_0, frozen BOL ECM (Step 2)  ->  V_hat(t) (derived output, after the MHE)
# 
# Temperature has no SoC information, so SoC estimate comes from prior and anchor (check step 2)
# [T, beta] is exactly step 1 and SoC is a 1-D problem
# gives exact checks in --self-test, set SOC_COUPLING = "joule_shape"
# or ANCHOR_R_SOURCE = "state" to make ME joint (ablation suite)

# Two V_hat products per CC:
# online: uses only what is know before charger switches to CV
# post: uses after CC end, SoC0 from MHE window
# Ablation 4

# Usage:
# python voltage_with_ecm.py --csv VAH01.csv --impedance VAH01_impedance.csv --out step2_results_bwd --soc0-ref backward
# python full_extended_mhe.py --csv VAH01.csv --handoff step2_results_bwd/VAH01_ecm_bol.json --self-test
# python full_extended_mhe.py --csv VAH01.csv --handoff step2_results_bwd/VAH01_ecm_bol.json --out results_step3
# python full_extended_mhe.py --csv VAH01.csv --handoff step2_results_bwd/VAH01_ecm_bol.json --ablations-only
# (without --handoff: step2_results/<cell>_ecm_bol.json, i.e. Step 2's default output folder)
# Flags that change the estimator: --soc0-mode, --no-anchor, --anchor-r, --coupling, --linearize,
# --sigma-plan-ah, --r-scale, --split, --fade-mode. Everything else is a module-level setting below.

# [PAPER] means from Ahuja et al 2026
# [DATA] means from the dataset 
# [ASSUMED] ambiguous in both data and paper


from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar, curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import base as s1  # noqa: E402  (importing it also silences do-mpc's optional-feature warnings)
import voltage_with_ecm as s2  # noqa: E402
import casadi as ca  # noqa: E402
import do_mpc  # noqa: E402



# Settings (module-level constants; some are overridden from the CLI in main())

# Shared with Step 1
DT_S = None                     
N_HORIZON = 30                  
Q_T_K = 0.02                    
Q_BETA_KPS = 2e-4               
ARRIVAL_W_TB = 1e-6             
T_RISE_MAX_K = 60.0    
BETA_INIT_KPS = 1e-3     
BETA_BOOTSTRAP_UB = 0.1        
RF_OVER_R0 = 2.0               
N_BETA0_EVENTS = 5     
TAIL_FRAC = 0.25               
T_AMB_MODE = "exp_fit"         
ALPHA_MODE = "ls"             

# Charger and duty cycle 
CC_CURRENT_A = 3.0             
V_CV = 4.2                    

# SoC frame: which coulomb count defines the mission Ah (truth and duty-cycle plan) ---
SOC_FRAME = "auto"              # "auto": the frame Step 2's hand-off was calibrated in (provenance soc0_ref; "forward"
                                #         if absent) | "forward": 1 - (Ah out since the last full) / Q | "backward":
                                #         1 - (Ah put back to the end of this charge's CV) / Q.
                                # [DATA] On VAH01 the logged discharge integral exceeds the Ah put back by 0.073 Ah at 1 s
                                # discharge logging and by 0.15 Ah after the change to 10 s (around cycle 300); the cycler's
                                # QDischarge counter shows no such step. "backward" is the physically consistent frame, but
                                # the frozen ECM must be calibrated in the same frame (step2 --soc0-ref backward).

# SoC channels 
SOC0_MODE = "duty_cycle"        # "duty_cycle" | "uninformed" | "oracle" [DEV-ONLY]
SIGMA_SOC0 = 0.02               # [ASSUMED] prior std of SoC0 in duty_cycle mode (Q_fade error, plan deviation)
SIGMA_SOC0_ORACLE = 1e-3        # [ASSUMED] effectively pinned
SOC0_UNINFORMED = (0.35, 0.10)  # [DATA] CLAUDE.md: starting SoC 25-45 % -> (mean, std)
SIGMA_PLAN_AH = 0.0             # [ASSUMED] noise on mission Ah (plan vs actual). 0: the cycler ran the plan, so the
                                # dataset's mission Ah (in SOC_FRAME) is the plan, and duty_cycle then differs from
                                # oracle only through Q_fade vs Q_rpt.
Q_SOC = 1e-4                    # [ASSUMED] SoC process noise per step (one 30 s step at 1C moves SoC by ~0.8-1 %)
ARRIVAL_SOC = "filtering"       # [ASSUMED] SoC prior of each window: "filtering" = the filtered estimate
                                # x_{k-N|k-N} (no measurement counted twice) | "smoothing" = do-mpc's default
                                # x_{k-N|k-1}. Identical with the defaults; differs when channel 3 is on.
SOC_COUPLING = "none"           # "none" | "joule_shape" (beta+ = beta * R_BOL(SoC+) / R_BOL(SoC))
V_WINDOW_LO = 2.5               # [DATA] logged min/max cell voltage cutoffs
V_WINDOW_HI = 4.2
CC_END_ANCHOR = True
ANCHOR_R_SOURCE = "trailing"    # "trailing" (r from previous charges) | "state" (beta / beta_0) | "oracle" [DEV-ONLY]
SIGMA_ANCHOR_V = 0.008          # [ASSUMED] Step 2 tier B end-of-CC bias ~ +6 mV at BOL; the switch instant is logged,
                                # so there is no timing term
LINEARIZE_ANCHOR = False        # Ablation 3: first-order Taylor of the anchor around the prior's end-of-CC SoC
R_TRAIL_N = 25                  # [ASSUMED] causal trailing mean of beta_k / beta_0 over the previous 25 charges (window in
                                # charges, not processed events, so --stride does not change how far back it reaches)
R_SCALE_SOURCE = "event"        # resistance scale for the post product: "event" | "trailing" | "oracle" [DEV-ONLY]
SPLIT_MODE = "fixed"            # "fixed" (Ablation A) | "aging" (Ablation B: phi + linear trend of the per-RPT
                                # upper-step phi from Step 2 tier C; the lower step is SoC-confounded)
LUT_KIND = "bspline"            # casadi interpolant through Step 2's 201-point grid (falls back to "linear" if
                                # the bspline OCV is not monotone)

# Capacity fade (Q is not observable from temperature)
FADE_MODE = "fit"               # "fit": polynomial in cycle count of Q/Q_BOL (from --fade-csv, else this cell's RPTs,
                                #        in-sample) times this cell's Q_BOL
                                # "causal": linear fit over the RPTs before each event, extrapolated
                                # "rpt_interp": Q interpolated in time between RPTs 
FADE_DEG = 2                   

# Validation and outputs
CV_EXT_DT_S = 5.0               # [ASSUMED] step when continuing V_hat past the CC end to find the CV crossing
CV_EXTEND_MAX_S = 1800.0        # [ASSUMED]
EARLY_FRAC = 0.10               # [ASSUMED] "early charge" = first 10 % of the CC duration (hysteresis region)
SMOOTH_HALF_WINDOW = 25         # [PAPER] +/- 25 cycles, for plots
PROTOCOL_CHANGE_EVENTS = (336, 560)   # [DATA] approximate: discharge logging 1 s -> 10 s at ~336 (Step 1 numbering),
                                      # rest length and rest logging change through ~560 (Step 1)
ABLATION_STRIDE = 10            # [ASSUMED]
PLAN_NOISE_AH_ABLATION = 0.06   # [ASSUMED] ~2 % of Q, used by the "plan_noise" ablations
PROFILE_N_EVENTS = 12           # [ASSUMED] self-test 5
RANDOM_SEED = 0
MAKE_PLOTS = True

# Plot colours: validated categorical order (blue, orange, aqua, yellow) + neutral inks
C_POST, C_ONLINE, C_SOC, C_R = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_INK, C_INK2, C_MEAS, C_GRID = "#0b0b0b", "#52514e", "#8a8a8a", "#e6e6e6"

# 1. Small helpers
 
@contextlib.contextmanager
def override(**kwargs):
#    Temporarily change module-level settings (used by the ablation suite and the self-tests).
    g = globals()
    for k in kwargs:
        if k not in g or not k.isupper():
            raise KeyError(f"unknown setting {k}")
    old = {k: g[k] for k in kwargs}
    g.update(kwargs)
    try:
        yield
    finally:
        g.update(old)
 
 
def rmse(a, b=None):
    a = np.asarray(a, float) if b is None else np.asarray(a, float) - np.asarray(b, float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")
 
 
def nanmean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")
 
 
def centred_ma(x, half):
    return pd.Series(np.asarray(x, float)).rolling(2 * half + 1, center=True, min_periods=1).mean().to_numpy()
 
 
def settings_dict():
    keys = [k for k, v in list(globals().items())
            if k.isupper() and isinstance(v, (int, float, str, bool, type(None), tuple))]
    return {k: globals()[k] for k in keys}
 
 
# 2. Data: Step 1 events (thermal inputs) joined with Step 2 coulomb counting

 
def step1_config():
    return s1.Config(cc_current_A=CC_CURRENT_A, alpha_mode=ALPHA_MODE, t_amb_mode=T_AMB_MODE, dt_s=DT_S,
                     n_horizon=N_HORIZON, t_rise_max_K=T_RISE_MAX_K, rf_over_r0=RF_OVER_R0,
                     beta_bootstrap_ub=BETA_BOOTSTRAP_UB, q_T_K=Q_T_K, q_beta_Kps=Q_BETA_KPS,
                     arrival_weight=ARRIVAL_W_TB, beta_init_Kps=BETA_INIT_KPS, n_beta0_events=N_BETA0_EVENTS,
                     tail_frac=TAIL_FRAC)
 
 
def load_data(csv_path):
# One Step 1 event per 1C CC charge (first charge after each RPT excluded, as in Step 1), joined by CC start
# time to Step 2's segmentation for the mission Ah (both frames) and the sample indices.
# V/I are used here only to segment and, later, to validate.
    cfg1 = step1_config()
    cell = s2.load_cell(csv_path)
    ev2 = s2.find_mission_events(cell)
    blocks = s2.find_rpt_blocks(cell, ev2)            # also flags the first charge after each RPT
    fulls = s2.full_indices(ev2, blocks)
    df = pd.DataFrame({"time_s": cell["t"], "Ecell_V": cell["V"], "I_A": cell["I"],
                       "Temperature__C": cell["T"], "Ns": cell["ns"]})
    ev1, _ = s1.extract_events(df, cfg1)
    del df
    by_t0 = {float(e["t0"]): e for e in ev2}
    events, skipped = [], 0
    for e1 in ev1:
        e2 = by_t0.get(float(e1["t0"]))
        j = -1 if e2 is None else int(np.searchsorted(fulls, e2["start"])) - 1
        if e2 is None or e2["after_rpt"] or j < 0 or e2["cv_stop"] is None:
            skipped += 1
            continue
        a = int(e2["start"])
        b = a + len(e1["cc_t"])
        ah_out = float(cell["cum_ah"][fulls[j]] - cell["cum_ah"][a])            # Ah out since the last full charge
        ah_back = float(cell["cum_ah"][e2["cv_stop"] - 1] - cell["cum_ah"][a])   # Ah put back to the end of the CV
        events.append(dict(e1, k2=int(e2["k"]), a=a, b=b, ah_out=ah_out, ah_back=ah_back, v_cv_meas=e2["v_cv"]))
    if skipped:
        print(f"  note: {skipped} Step 1 event(s) could not be joined to Step 2's segmentation and were dropped")
    return dict(name=cell["name"], cell=cell, blocks=blocks, fulls=fulls, events=events, cfg1=cfg1)
 
 
def mission_ah(ev):
# Mission Ah in the SoC frame in use (truth and duty-cycle plan share it)
    if SOC_FRAME == "backward":
        return ev["ah_back"]
    if SOC_FRAME == "forward":
        return ev["ah_out"]
    raise ValueError(f"SOC_FRAME must be resolved to 'forward' or 'backward', got {SOC_FRAME}")
 
 
def select_events(events, stride=1, max_events=None):
# As Step 1: every stride-th event, but always the first N_BETA0_EVENTS
    sel = events[::max(1, stride)]
    if max_events:
        sel = sel[:max_events]
    head = events[:N_BETA0_EVENTS]
    return head + [e for e in sel if e["event"] >= head[-1]["event"] + 1]
 
 

# 3. Capacity, lookup tables, ECM, split trend

class Capacity:
# Q_rpt(t): RPT capacities interpolated in time (oracle, Step 2 tier D convention) [DEV-ONLY].
# Q_fade(k): the deployable fade curve, from training data (--fade-csv) or this cell's RPTs (in-sample).
 
    def __init__(self, blocks, fade_csv=None):
        self.tq = np.array([b["t_full"] for b in blocks], float)
        self.qq = np.array([b["q"] for b in blocks], float)
        self.cq = np.array([b["cycle"] for b in blocks], float)
        self.q_bol = float(self.qq[0])
        if fade_csv:
            f = pd.read_csv(fade_csv).sort_values("cycle")
            c, q = f["cycle"].to_numpy(float), f["q"].to_numpy(float)
            self.source, self.in_sample = f"training file {fade_csv}", False
        else:
            c, q = self.cq, self.qq
            self.source, self.in_sample = "this cell's RPTs (in-sample)", True
        self.coef = np.polyfit(c, q / q[0], FADE_DEG)       # normalised: Q/Q_BOL vs cycle
 
    def q_rpt(self, t0):
        return float(np.interp(t0, self.tq, self.qq))
 
    def q_fade(self, k2, t0):
        if FADE_MODE == "rpt_interp":
            return self.q_rpt(t0)
        if FADE_MODE == "causal":
            m = self.tq < t0
            if m.sum() >= 2:
                return float(np.polyval(np.polyfit(self.cq[m], self.qq[m], 1), k2))
            return float(self.qq[m][-1]) if m.any() else self.q_bol
        if FADE_MODE == "fit":
            return float(self.q_bol * np.polyval(self.coef, k2))
        raise ValueError(FADE_MODE)
 
 
class Luts:
# casadi lookup tables for OCV(SoC) and R_total_BOL(SoC) from Step 2's hand-off grid
 
    def __init__(self, handoff):
        m = handoff["model"]
        g = np.asarray(m["soc_grid"], float)
        ocv, rt = np.asarray(m["ocv_V"], float), np.asarray(m["r_total_ohm"], float)
        fine = np.linspace(0.0, 1.0, 5001)
        self.kind = LUT_KIND
        for kind in (LUT_KIND, "linear"):
            self.OCV = ca.interpolant("ocv", kind, [g.tolist()], ocv.tolist())
            self.RBOL = ca.interpolant("rbol", kind, [g.tolist()], rt.tolist())
            o = np.array(self.OCV(fine)).ravel()
            self.kind = kind
            if np.all(np.diff(o) > 0):
                break
            print(f"  WARNING: {kind} OCV interpolant is not monotone; falling back to linear")
        self.ocv_monotone = bool(np.all(np.diff(o) > 0))
        self.max_dev_ocv_mV = 1e3 * float(np.max(np.abs(o - np.interp(fine, g, ocv))))
        self.max_dev_r_mOhm = 1e3 * float(np.max(np.abs(np.array(self.RBOL(fine)).ravel() - np.interp(fine, g, rt))))
        s, r = ca.SX.sym("s"), ca.SX.sym("r")
        v = self.OCV(s) + CC_CURRENT_A * self.RBOL(s) * r
        self.vq = ca.Function("vq", [s, r], [v, ca.jacobian(v, s)])    # quasi-steady CC voltage and dV/dSoC
        info = handoff.get("resistance_info", {})
        self.rcc_range = tuple(info.get("rcc_soc_range", (np.nan, np.nan)))
 
    def rbol(self, s):
        return np.array(self.RBOL(np.atleast_1d(np.asarray(s, float)))).ravel()
 
    def vq_num(self, s, r):
        v, dv = self.vq(float(s), float(r))
        return float(v), float(dv)
 
 
def load_ecm(handoff):
    m = handoff["model"]
    return s2.EcmModel(m["soc_grid"], m["ocv_V"], m["r_total_ohm"], m["phi"], m["tau_s"], m["soc_valid"],
                       m["q_bol_Ah"], m["t_ref_C"])
 
 
def load_phi_trend(path):
# Ablation B: linear trend of the per-RPT upper-step phi (Step 2 tier C) vs cycle, as a shift from cycle 0
    if not path or not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    ok = df["ok"].astype(bool) & (df["soc"] > 0.6)
    if ok.sum() < 2:
        return None, None
    coef = np.polyfit(df.loc[ok, "cycle"].to_numpy(float), df.loc[ok, "phi"].to_numpy(float), 1)
    return (lambda c: float(np.polyval(coef, c) - np.polyval(coef, 0.0))), coef.tolist()
 
 
# 4. The extended MHE (do-mpc)
 
class ExtendedMHE:
# MHE for x = [T, beta, SoC]. Discrete model, exact zero-order hold (as Step 1):
# T+    = e T + (1 - e) T_amb + (1 - e)/alpha * beta + w_T   
# e = exp(-alpha dt)
# beta+ = beta (* R_BOL(SoC + dSoC) / R_BOL(SoC) with "joule_shape") + w_beta
# SoC+  = SoC + dSoC + w_SoC
# y_T = T + v_T
# y_A = anchor_on * (V_qs(SoC + ds_end) - V_target) + v_A            
# pseudo-measurement, fed y_A = 0
#
# V_qs = OCV(s) + I * R_BOL(s) * r is the quasi-steady CC voltage (V1 = Rp I: tau_eff ~ 19 s, far shorter
# than the time since CC start). anchor_on = 1 only at the node of the last CC sample.

# stage j of the window maps node j -> j+1 with
# tvp[j], and measurement j is applied to node j+1, so tvp[j] belongs to sample k - (N-1) + j when sample k
# is the newest. The template has N+1 entries; the NLP uses the first N. Before N samples exist, do-mpc
# pads the window by repeating the first measurement: dSoC is 0 on those padding stages, otherwise the SoC
# prior (arrival cost on node 0) would be shifted by up to N steps of charge.
 
    TVP = ("T_amb", "dsoc", "anchor_on", "anchor_ds", "v_target", "r_anchor", "s_lin", "v_lin", "g_lin")
 
    def __init__(self, luts, alpha, dt, beta_ub, sigma_T, sigma_soc0, beta_0, rbol_ref):
        self.N, self.dt = N_HORIZON, dt
        self.anchor, self.coupling, self.r_source = CC_END_ANCHOR, SOC_COUPLING, ANCHOR_R_SOURCE
        self.linear, self.arrival_soc = LINEARIZE_ANCHOR, ARRIVAL_SOC
        self.beta_0, self.rbol_ref = float(beta_0), float(rbol_ref)
        self.sigma_soc0 = float(sigma_soc0)
        if self.coupling not in ("none", "joule_shape"):
            raise ValueError(SOC_COUPLING)
        if self.r_source not in ("trailing", "state", "oracle"):
            raise ValueError(ANCHOR_R_SOURCE)
        OCV, RBOL = luts.OCV, luts.RBOL
 
        model = do_mpc.model.Model("discrete", "SX")
        T = model.set_variable("_x", "T")
        beta = model.set_variable("_x", "beta")
        soc = model.set_variable("_x", "soc")
        p = {nm: model.set_variable("_tvp", nm) for nm in self.TVP}
 
        ead = float(np.exp(-alpha * dt))
        model.set_rhs("T", ead * T + (1.0 - ead) * p["T_amb"] + (1.0 - ead) / alpha * beta, process_noise=True)
        if self.coupling == "joule_shape":
            model.set_rhs("beta", beta * RBOL(soc + p["dsoc"]) / RBOL(soc), process_noise=True)
        else:
            model.set_rhs("beta", beta, process_noise=True)
        model.set_rhs("soc", soc + p["dsoc"], process_noise=True)
        model.set_meas("T_meas", T, meas_noise=True)
        if self.anchor:
            s_end = soc + p["anchor_ds"]
            if self.linear:      # linear in the states: Taylor in SoC around s_lin, resistance term at s_lin
                r_mod = self._r_scale(beta, p["s_lin"], p["r_anchor"], RBOL)
                v_end = (p["v_lin"] + p["g_lin"] * (s_end - p["s_lin"])
                         + CC_CURRENT_A * RBOL(p["s_lin"]) * (r_mod - p["r_anchor"]))
            else:
                v_end = OCV(s_end) + CC_CURRENT_A * RBOL(s_end) * self._r_scale(beta, soc, p["r_anchor"], RBOL)
            model.set_meas("V_anchor", p["anchor_on"] * (v_end - p["v_target"]), meas_noise=True)
        model.setup()
 
        mhe = do_mpc.estimator.MHE(model)
        mhe.settings.n_horizon = self.N
        mhe.settings.t_step = dt
        mhe.settings.meas_from_data = True
        mhe.settings.store_full_solution = False
        mhe.settings.store_lagr_multiplier = False
        mhe.settings.nlpsol_opts = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0}
 
        P_v = np.diag([1.0 / sigma_T ** 2] + ([1.0 / SIGMA_ANCHOR_V ** 2] if self.anchor else []))
        P_w = np.diag([1.0 / Q_T_K ** 2, 1.0 / Q_BETA_KPS ** 2, 1.0 / Q_SOC ** 2])
        # Arrival cost: negligible on T and beta (paper Eq. 9, as Step 1), real on SoC: it IS the SoC0 prior.
        P_x = np.diag([ARRIVAL_W_TB / sigma_T ** 2, ARRIVAL_W_TB / Q_BETA_KPS ** 2, 1.0 / self.sigma_soc0 ** 2])
        mhe.set_default_objective(P_x=P_x, P_v=P_v, P_w=P_w)
 
        mhe.bounds["lower", "_x", "beta"] = 0.0                  # Eq. 10 (Step 1)
        mhe.bounds["upper", "_x", "beta"] = beta_ub
        mhe.bounds["lower", "_x", "soc"] = 0.0
        mhe.bounds["upper", "_x", "soc"] = 1.0
        mhe.set_nl_cons("T_lower", p["T_amb"] - T, ub=0.0)
        mhe.set_nl_cons("T_upper", T - p["T_amb"] - T_RISE_MAX_K, ub=0.0)
 
        tvp = mhe.get_tvp_template()
        n_tvp = len(self.TVP)
        names = [lab.strip("[]").split(",")[2] for lab in tvp.labels()[:n_tvp]]
        self._col = {nm: names.index(nm) for nm in self.TVP}
        self._rows = tvp.master.shape[0] // n_tvp                # N + 1
        self._base = np.zeros((self._rows, n_tvp))
        self.ctx = dict(k=0, dsoc=0.0, i_anchor=-1)
 
        def tvp_fun(_t_now):
            # Own sample counter, not do-mpc's clock (reset_history does not need to reset it then).
            arr = self._base.copy()
            i = self.ctx["k"] - (self.N - 1) + np.arange(self._rows)    # sample index of each stage's end node
            arr[:, self._col["dsoc"]] = np.where(i >= 1, self.ctx["dsoc"], 0.0)
            arr[:, self._col["anchor_on"]] = (i == self.ctx["i_anchor"]).astype(float)
            tvp.master = ca.DM(arr.ravel())
            return tvp
 
        mhe.set_tvp_fun(tvp_fun)
        mhe.setup()
        self.mhe = mhe
 
    def _r_scale(self, beta, soc_ref, r_anchor, RBOL):
        """Resistance scale r in the anchor, R_total = R_BOL(SoC) * r."""
        if self.r_source != "state":
            return r_anchor                               # trailing / oracle: from outside, as a tvp
        r = beta / self.beta_0                            # Step 1's R_ratio_T
        if self.coupling == "joule_shape":                # beta already carries the R_BOL(SoC) shape
            r = r * self.rbol_ref / RBOL(soc_ref)
        return r
 
    def run_event(self, T_cc, T_amb, soc0_prior, dsoc, anc):
        """Feed one CC event (uniform grid) through the MHE. anc: dict(i, ds, v_target, r, s_lin, v_lin, g_lin).
        Returns filtered (causal) and smoothed (last-window) trajectories."""
        mhe, N, n = self.mhe, self.N, len(T_cc)
        c = self._col
        self._base[:] = 0.0
        self._base[:, c["T_amb"]] = T_amb
        self._base[:, c["anchor_ds"]] = anc["ds"]
        self._base[:, c["v_target"]] = anc["v_target"]
        self._base[:, c["r_anchor"]] = anc["r"]
        self._base[:, c["s_lin"]] = anc["s_lin"]
        self._base[:, c["v_lin"]] = anc["v_lin"]
        self._base[:, c["g_lin"]] = anc["g_lin"]
        self.ctx.update(k=0, dsoc=float(dsoc), i_anchor=int(anc["i"]) if self.anchor else -1)
 
        mhe.reset_history()
        x0 = np.array([[max(float(T_cc[0]), T_amb)], [BETA_INIT_KPS], [float(soc0_prior)]])
        mhe.x0 = x0
        mhe.set_initial_guess()
        xs = mhe._x_scaling.cat.full()
        mhe.opt_x_num["_x", :, :] = x0 / xs                     # re-seed every node (as Step 1)
        mhe.opt_x_num["_w", :] = 0.0
        mhe.opt_x_num["_v", :] = 0.0
 
        filt = np.empty((n, 3))
        n_fail = 0
        for k in range(n):
            self.ctx["k"] = k
            if self.arrival_soc == "filtering":
                # make_step takes the arrival prior from node 1 of the previous solution; overwrite its SoC part
                # with x_{k-N|k-N} (only the warm start of that node changes otherwise).
                i0 = k - N
                mhe.opt_x_num["_x", 1, -1, "soc"] = (filt[i0, 2] if i0 >= 0 else soc0_prior) / xs[2, 0]
            y = np.array([[T_cc[k]], [0.0]]) if self.anchor else np.array([[T_cc[k]]])
            filt[k] = mhe.make_step(y).ravel()
            if not mhe.solver_stats.get("success", True):
                n_fail += 1
 
        win = np.hstack([mhe.opt_x_num["_x", j, -1].full() for j in range(N + 1)]).T * xs.T   # (N+1, 3)
        idx = n - 1 - N + np.arange(N + 1)                       # node j <-> sample n-1-N+j
        ok = idx >= 0
        soc_s = np.full(n, np.nan)
        beta_s = filt[:, 1].copy()
        soc_s[idx[ok]] = win[ok, 2]
        beta_s[idx[ok]] = win[ok, 1]
        first = int(idx[ok][0])
        for i in range(first - 1, -1, -1):                       # before the window: no new information
            soc_s[i] = soc_s[i + 1] - dsoc
        w_soc = float(np.max(np.abs(np.diff(soc_s[first:]) - dsoc))) if n - first > 1 else 0.0
        soc_pre = filt[:, 2].copy()                              # online: never uses the anchor
        if self.anchor and int(anc["i"]) == n - 1 and n >= 2:
            soc_pre[-1] = filt[-2, 2] + dsoc
        return dict(T_hat=filt[:, 0], beta=filt[:, 1], soc_filt=filt[:, 2], soc_pre=soc_pre,
                    soc_smooth=soc_s, beta_smooth=beta_s, soc0_hat=float(soc_s[0]), n_fail=n_fail, w_soc_max=w_soc)
 
 

# 5. Anchor helpers, SoC0 prior, scalar cross-check

 
def anchor_voltage(luts, s_end, r, anc=None, linear=False):
# Numerical twin of the model's anchor voltage (the MHE's expression, evaluated outside casadi)
    if not linear:
        return luts.vq_num(s_end, r)[0]
    return (anc["v_lin"] + anc["g_lin"] * (s_end - anc["s_lin"])
            + CC_CURRENT_A * float(luts.rbol(anc["s_lin"])[0]) * (r - anc["r"]))
 
 
def soc0_scalar(luts, prior, sigma0, dsoc, anc, linear=False):
# Self-test 2: with the block-diagonal defaults the MHE's SoC0 must equal this 1-D solution
    off = anc["i"] * dsoc + anc["ds"]
 
    def J(s0):
        dv = anchor_voltage(luts, s0 + off, anc["r"], anc, linear) - anc["v_target"]
        return ((s0 - prior) / sigma0) ** 2 + (dv / SIGMA_ANCHOR_V) ** 2
 
    return float(minimize_scalar(J, bounds=(0.0, 1.0), method="bounded", options=dict(xatol=1e-9)).x)
 
 
def sigma_soc0_for_mode():
    return {"oracle": SIGMA_SOC0_ORACLE, "duty_cycle": SIGMA_SOC0, "uninformed": SOC0_UNINFORMED[1]}[SOC0_MODE]
 
 
def soc0_prior_for(ev, q_k, soc0_true):
    if SOC0_MODE == "oracle":                                    # [DEV-ONLY] H2's "exact SoC0"
        return float(soc0_true)
    if SOC0_MODE == "duty_cycle":                                # deployment: plan Ah + fade curve
        ah = mission_ah(ev)
        if SIGMA_PLAN_AH > 0:
            ah += np.random.default_rng([RANDOM_SEED, int(ev["event"])]).normal(0.0, SIGMA_PLAN_AH)
        return float(1.0 - ah / q_k)                             # [ASSUMED] the previous CV ended at SoC 1
    if SOC0_MODE == "uninformed":
        return float(SOC0_UNINFORMED[0])
    raise ValueError(SOC0_MODE)
 
 

# 6. Derived output V(t) and validation helpers

 
def ecm_voltage(ecm, t, soc, r, phi_shift=0.0, q=None, v_cv=None):
# V_hat(t) from a SoC trajectory and a resistance scale with Step 2's frozen BOL ECM, at the setpoint current
# (never the measured one). v1_0 = 0: the pre-charge rest (>= 600 s) is much longer than tau_eff (19 s)
# With v_cv: also the time V_hat reaches v_cv, continuing the charge past t[-1] at the same current
    t, soc = np.asarray(t, float), np.asarray(soc, float)
    if v_cv is None:
        return ecm.simulate(t, np.full(len(t), CC_CURRENT_A), soc, v1_0=0.0, r_scale=r, phi_shift=phi_shift), np.nan
    t_ext = t[-1] + np.arange(CV_EXT_DT_S, CV_EXTEND_MAX_S + 1e-9, CV_EXT_DT_S)
    tt = np.r_[t, t_ext]
    ss = np.r_[soc, soc[-1] + CC_CURRENT_A * (t_ext - t[-1]) / 3600.0 / q]
    vv = ecm.simulate(tt, np.full(len(tt), CC_CURRENT_A), ss, v1_0=0.0, r_scale=r, phi_shift=phi_shift)
    hit = np.flatnonzero(vv >= v_cv)
    if not hit.size:
        tc = np.nan
    elif hit[0] == 0:
        tc = 0.0
    else:
        k = int(hit[0])
        tc = float(tt[k - 1] + (v_cv - vv[k - 1]) * (tt[k] - tt[k - 1]) / (vv[k] - vv[k - 1]))
    return vv[:len(t)], tc
 
 
def fit_r_oracle(ecm, t, soc, V, phi_shift):
# best single resistance scale for this charge with the true SoC (Step 2 tier D's s(c))
    def f(z):
        return ecm_voltage(ecm, t, soc, z[0], phi_shift)[0] - V
    return float(least_squares(f, [1.0], bounds=([0.2], [5.0])).x[0])
 
 

# 7. Thermal calibration (Step 1, unchanged) and per-event inputs

def calibrate_thermal(D, luts, cap):
#    alpha, sigma_T, T_amb(BOL) from the first five rests and beta_0 from Step 1's pass A, with Step 1's code.
    cfg1, events = D["cfg1"], D["events"]
    ainfo = s1.calibrate_alpha([events], cfg1)
    dts = [np.median(np.diff(e["cc_t"])) for e in events]
    dt = DT_S or float(np.round(np.median(dts), 3))
    sigma_T = cfg1.sigma_T_K or ainfo["sigma_T"]
    head = events[:N_BETA0_EVENTS]
    est_A = s1.ThermalMHE(ainfo["alpha"], dt, BETA_BOOTSTRAP_UB, sigma_T, cfg1)
    tab_A, _ = s1.run_mhe_over_events(est_A, head, cfg1, t_amb_bol=ainfo["t_amb_bol"][0])
    beta_0 = float(tab_A["beta_k"].mean())
    # R_BOL reference for "joule_shape": mean R_BOL over the last quarter of the BOL charges along the prior path,
    # so that beta_k / beta_0 is still ~ r with the coupling on
    refs = []
    for ev in head:
        q = cap.q_fade(ev["k2"], ev["t0"])
        n = len(s1.resample_uniform(ev["cc_t"], ev["cc_T"], dt))
        soc = 1.0 - mission_ah(ev) / q + CC_CURRENT_A * dt * np.arange(n) / 3600.0 / q
        refs.append(float(np.mean(luts.rbol(soc[int(np.floor(n * (1 - TAIL_FRAC))):]))))
    return dict(alpha=float(ainfo["alpha"]), alpha_ls=float(ainfo["alpha_ls"]), sigma_T=float(sigma_T), dt=dt,
                t_amb_bol=float(ainfo["t_amb_bol"][0]), beta_0=beta_0, beta_f=RF_OVER_R0 * beta_0,
                rbol_ref=float(np.mean(refs)))
 
 
def prepare_event(ev, D, therm, luts, cap, r_trail, t_amb_last, phi_trend=None):
# Everything the MHE and the validation need for one event
    cell, dt = D["cell"], therm["dt"]
    t_amb = s1.infer_t_amb(ev["rest"], D["cfg1"], fallback=t_amb_last, t_amb_bol=therm["t_amb_bol"])
    if not np.isfinite(t_amb):
        t_amb = float(ev["cc_T"][0])
    T_cc = s1.resample_uniform(ev["cc_t"], ev["cc_T"], dt)
    n = len(T_cc)
    t_nat = np.asarray(ev["cc_t"], float)
    t_end = float(t_nat[-1]) # CC end = the logged switch instant
    a, b = ev["a"], ev["b"]
    q_k = cap.q_fade(ev["k2"], ev["t0"])
    q_true = cap.q_rpt(ev["t0"])
    dsoc = CC_CURRENT_A * dt / 3600.0 / q_k
    soc0_true = 1.0 - mission_ah(ev) / q_true # [DEV-ONLY] oracle (Step 2 tier D convention)
    soc_true = soc0_true + (cell["cum_ah"][a:b] - cell["cum_ah"][a]) / q_true
    V_meas = cell["V"][a:b]
    phi_shift = phi_trend(ev["k2"]) if (SPLIT_MODE == "aging" and phi_trend is not None) else 0.0
    r_or = fit_r_oracle(D["ecm"], t_nat, soc_true, V_meas, phi_shift)
    soc0_prior = soc0_prior_for(ev, q_k, soc0_true)
    r_anc = r_or if ANCHOR_R_SOURCE == "oracle" else r_trail
    ds_end = CC_CURRENT_A * (t_end - (n - 1) * dt) / 3600.0 / q_k # last grid node -> switch instant
    s_lin = soc0_prior + (n - 1) * dsoc + ds_end
    v_lin, g_lin = luts.vq_num(s_lin, r_anc)
    anc = dict(i=n - 1, ds=ds_end, v_target=V_CV, r=r_anc, s_lin=s_lin, v_lin=v_lin, g_lin=g_lin)
    return dict(t_amb=float(t_amb), T_cc=T_cc, n=n, t_nat=t_nat, t_end=t_end, q_k=q_k, q_true=q_true, dsoc=dsoc,
                soc0_true=float(soc0_true), soc_true=soc_true, V_meas=V_meas, phi_shift=phi_shift, r_or=r_or,
                soc0_prior=soc0_prior, anc=anc)
 
 

# 8. One configuration over a list of events
 
def trailing_r(r_hist, event):
# Causal mean of beta_k / beta_0 over the charges in [event - R_TRAIL_N, event); the latest one if the window
# holds none (sparse --stride), 1.0 before any
    if not r_hist:
        return 1.0
    win = [r for e, r in r_hist if event - R_TRAIL_N <= e < event]
    return float(np.mean(win)) if win else float(r_hist[-1][1])
 
def run_config(D, therm, luts, cap, phi_trend, events, keep_traces=(), label=""):
    if SPLIT_MODE == "aging" and phi_trend is None:
        raise RuntimeError("SPLIT_MODE = 'aging' needs Step 2's tier C file (--tierc)")
    ecm = D["ecm"]
    sigma0 = sigma_soc0_for_mode()
    est = ExtendedMHE(luts, therm["alpha"], therm["dt"], therm["beta_f"], therm["sigma_T"], sigma0,
                      therm["beta_0"], therm["rbol_ref"])
    block_diag = CC_END_ANCHOR and ANCHOR_R_SOURCE != "state" and SOC_COUPLING == "none"
    dt, beta_0 = therm["dt"], therm["beta_0"]
    rows, traces, r_hist = [], {}, []
    t_amb_last = np.nan
    t_start, n_steps = time.time(), 0
    for n_ev, ev in enumerate(events):
        r_trail = trailing_r(r_hist, ev["event"])
        P = prepare_event(ev, D, therm, luts, cap, r_trail, t_amb_last, phi_trend)
        t_amb_last = P["t_amb"]
        n, q_k, dsoc, anc, t_nat = P["n"], P["q_k"], P["dsoc"], P["anc"], P["t_nat"]
        out = est.run_event(P["T_cc"], P["t_amb"], P["soc0_prior"], dsoc, anc)
        n_steps += n
 
        beta_k = s1.beta_k_from_trace(out["beta"], TAIL_FRAC)          # Step 1 rule, unchanged
        r_event = beta_k / beta_0
        r_hist.append((ev["event"], r_event))
        r_post = {"event": r_event, "trailing": r_trail, "oracle": P["r_or"]}[R_SCALE_SOURCE]
 
        # SoC trajectories on the native CC timestamps
        kk = np.clip(np.floor(t_nat / dt + 1e-6).astype(int), 0, n - 1)
        soc_on = out["soc_pre"][kk] + CC_CURRENT_A * (t_nat - kk * dt) / 3600.0 / q_k
        soc_post = out["soc0_hat"] + CC_CURRENT_A * t_nat / 3600.0 / q_k
        ph = P["phi_shift"]
        V_on, tcv_on = ecm_voltage(ecm, t_nat, soc_on, r_trail, ph, q_k, V_CV)
        V_post, tcv_post = ecm_voltage(ecm, t_nat, soc_post, r_post, ph, q_k, V_CV)
        V_floor = ecm_voltage(ecm, t_nat, P["soc_true"], P["r_or"], ph)[0]      # [DEV-ONLY] attribution
        V_r_only = ecm_voltage(ecm, t_nat, P["soc_true"], r_post, ph)[0]
        V_soc_only = ecm_voltage(ecm, t_nat, soc_post, P["r_or"], ph)[0]
        V = P["V_meas"]
        early = t_nat <= EARLY_FRAC * P["t_end"]
        if early.sum() < 2:
            early[:2] = True
 
        s_end_hat = float(out["soc_smooth"][-1] + anc["ds"])
        if CC_END_ANCHOR:
            if ANCHOR_R_SOURCE == "state":
                b_end, s_last = float(out["beta_smooth"][-1]), float(out["soc_smooth"][-1])
                r_used = b_end / beta_0
                if SOC_COUPLING == "joule_shape":
                    s_ref = anc["s_lin"] if LINEARIZE_ANCHOR else s_last
                    r_used *= therm["rbol_ref"] / float(luts.rbol(s_ref)[0])
            else:
                r_used = anc["r"]
            resid = 1e3 * (anchor_voltage(luts, s_end_hat, r_used, anc, LINEARIZE_ANCHOR) - anc["v_target"])
        else:
            resid = np.nan
        soc0_sc = (soc0_scalar(luts, P["soc0_prior"], sigma0, dsoc, anc, LINEARIZE_ANCHOR)
                   if block_diag else np.nan)
        lo, hi = luts.rcc_range
        rows.append(dict(
            event=ev["event"], k2=ev["k2"], t0_s=ev["t0"], has_rest=ev["rest"] is not None, T_amb=P["t_amb"],
            cc_samples=n, cc_dur_s=P["t_end"], Q_k=q_k, Q_rpt=P["q_true"], ah_out=ev["ah_out"], ah_back=ev["ah_back"],
            soc0_true=P["soc0_true"], soc0_prior=P["soc0_prior"], soc0_hat=out["soc0_hat"], soc0_scalar=soc0_sc,
            soc_end_true=float(P["soc_true"][-1]), soc_end_hat=s_end_hat,
            anchor_extrap=bool(np.isfinite(lo) and not (lo <= s_end_hat <= hi)),
            beta_k=beta_k, r_event=r_event, r_trail=r_trail, r_post=r_post, r_anchor=anc["r"], r_oracle=P["r_or"],
            phi_shift=ph, anchor_resid_mV=resid,
            rmse_online_mV=1e3 * rmse(V_on, V), rmse_post_mV=1e3 * rmse(V_post, V),
            rmse_online_early_mV=1e3 * rmse(V_on[early], V[early]),
            rmse_post_early_mV=1e3 * rmse(V_post[early], V[early]),
            end_err_online_mV=1e3 * float(V_on[-1] - V[-1]), end_err_post_mV=1e3 * float(V_post[-1] - V[-1]),
            cv_dt_online_s=tcv_on - P["t_end"], cv_dt_post_s=tcv_post - P["t_end"],
            rmse_floor_mV=1e3 * rmse(V_floor, V), rmse_r_only_mV=1e3 * rmse(V_r_only, V),
            rmse_soc_only_mV=1e3 * rmse(V_soc_only, V),
            fit_rmse_K=rmse(out["T_hat"], P["T_cc"]), w_soc_max=out["w_soc_max"], n_solver_fail=out["n_fail"]))
        if ev["event"] in keep_traces:
            traces[ev["event"]] = dict(t=t_nat, V=V, V_on=V_on, V_post=V_post, V_floor=V_floor, soc_true=P["soc_true"],
                                       soc_on=soc_on, soc_post=soc_post, t_grid=np.arange(n) * dt, T=P["T_cc"],
                                       T_hat=out["T_hat"], beta=out["beta"])
        if (n_ev + 1) % 100 == 0:
            el = time.time() - t_start
            print(f"    {label}{n_ev + 1}/{len(events)} events ({el:.0f} s, {1e3 * el / n_steps:.1f} ms/step)", flush=True)
    el = time.time() - t_start
    return pd.DataFrame(rows), traces, dict(runtime_s=el, ms_per_step=1e3 * el / max(n_steps, 1))
 
 
def summarise(tab):
    e_prior = (tab["soc0_prior"] - tab["soc0_true"]).to_numpy()
    e_hat = (tab["soc0_hat"] - tab["soc0_true"]).to_numpy()
    dr = (tab["r_post"] - tab["r_oracle"]).to_numpy()
    n10 = max(1, len(tab) // 10)
    first, last = tab.head(n10), tab.tail(n10)
    corr = float(np.corrcoef(e_hat, dr)[0, 1]) if np.std(e_hat) > 1e-9 and np.std(dr) > 1e-9 else np.nan
    m = lambda c, d=tab: nanmean(d[c])  # noqa: E731
    out = _summarise_mV(tab, e_prior, e_hat, dr, first, last, corr, m)
    # NRMSE: every V-RMSE in mV also reported as % of the operating window (one fixed denominator, then averaged)
    win_mV = 1e3 * (V_WINDOW_HI - V_WINDOW_LO)
    out.update({k[:-3] + "_pct": 100.0 * v / win_mV for k, v in list(out.items())
                if k.startswith(("rmse_", "first10_rmse_", "last10_rmse_", "h2_")) and k.endswith("_mV")})
    return out
 
 
def _summarise_mV(tab, e_prior, e_hat, dr, first, last, corr, m):
    return dict(
        events=int(len(tab)),
        rmse_online_mV=m("rmse_online_mV"), rmse_post_mV=m("rmse_post_mV"), rmse_floor_mV=m("rmse_floor_mV"),
        rmse_r_only_mV=m("rmse_r_only_mV"), rmse_soc_only_mV=m("rmse_soc_only_mV"),
        rmse_online_early_mV=m("rmse_online_early_mV"), rmse_post_early_mV=m("rmse_post_early_mV"),
        h2_soc_part_mV=nanmean(tab["rmse_soc_only_mV"] - tab["rmse_floor_mV"]),
        h2_r_part_mV=nanmean(tab["rmse_r_only_mV"] - tab["rmse_floor_mV"]),
        soc0_prior_err_rms_pct=100 * rmse(e_prior), soc0_prior_err_mean_pct=100 * nanmean(e_prior),
        soc0_hat_err_rms_pct=100 * rmse(e_hat), soc0_hat_err_mean_pct=100 * nanmean(e_hat),
        end_err_online_mean_mV=m("end_err_online_mV"), cv_dt_online_mean_s=m("cv_dt_online_s"),
        cv_dt_online_mean_abs_s=nanmean(np.abs(tab["cv_dt_online_s"])),
        cv_online_not_reached=int(tab["cv_dt_online_s"].isna().sum()),
        end_err_post_mean_mV=m("end_err_post_mV"), cv_dt_post_mean_abs_s=nanmean(np.abs(tab["cv_dt_post_s"])),
        conflation_corr_soc0err_vs_rerr=corr,
        first10_rmse_post_mV=m("rmse_post_mV", first), last10_rmse_post_mV=m("rmse_post_mV", last),
        first10_rmse_online_mV=m("rmse_online_mV", first), last10_rmse_online_mV=m("rmse_online_mV", last),
        first10_rmse_floor_mV=m("rmse_floor_mV", first), last10_rmse_floor_mV=m("rmse_floor_mV", last),
        first10_soc0_hat_err_mean_pct=100 * nanmean((first["soc0_hat"] - first["soc0_true"])),
        last10_soc0_hat_err_mean_pct=100 * nanmean((last["soc0_hat"] - last["soc0_true"])),
        first10_r_oracle=m("r_oracle", first), last10_r_oracle=m("r_oracle", last),
        first10_r_event=m("r_event", first), last10_r_event=m("r_event", last),
        anchor_extrap_frac=float(tab["anchor_extrap"].mean()) if CC_END_ANCHOR else np.nan,
        anchor_resid_mean_mV=m("anchor_resid_mV"),
        scalar_check_max_abs=float(np.nanmax(np.abs(tab["soc0_hat"] - tab["soc0_scalar"])))
        if tab["soc0_scalar"].notna().any() else np.nan,
        solver_fail_steps=int(tab["n_solver_fail"].sum()), w_soc_max=float(tab["w_soc_max"].max()))

 
# 9. Ablation suite

def ablation_list():
    pn = PLAN_NOISE_AH_ABLATION
    return [
        ("default", "duty-cycle prior + CC-end anchor (anchor r from previous charges)", {}),
        ("no_anchor", "duty-cycle prior only", dict(CC_END_ANCHOR=False)),
        ("oracle_soc0", "H2: exact SoC0, no anchor [DEV-ONLY]", dict(SOC0_MODE="oracle", CC_END_ANCHOR=False)),
        ("oracle_r", "H2: exact resistance scale in the post product [DEV-ONLY]", dict(R_SCALE_SOURCE="oracle")),
        ("r_trailing", "post product with the trailing resistance scale", dict(R_SCALE_SOURCE="trailing")),
        ("plan_noise", f"prior from a noisy plan (+/-{pn} Ah), no anchor", dict(SIGMA_PLAN_AH=pn, CC_END_ANCHOR=False)),
        ("plan_noise+anchor", "noisy plan + anchor", dict(SIGMA_PLAN_AH=pn)),
        ("plan_noise+anchor_lin", "Ablation 3: noisy plan + linearised anchor", dict(SIGMA_PLAN_AH=pn, LINEARIZE_ANCHOR=True)),
        ("uninformed", "SoC0 ~ N(0.35, 0.10), no anchor", dict(SOC0_MODE="uninformed", CC_END_ANCHOR=False)),
        ("uninformed+anchor", "uninformed prior + anchor", dict(SOC0_MODE="uninformed")),
        ("uninformed+anchor_lin", "Ablation 3: uninformed + linearised anchor", dict(SOC0_MODE="uninformed", LINEARIZE_ANCHOR=True)),
        ("anchor_r_state", "Ablation 4: noisy plan, anchor r = beta / beta_0 (joint)", dict(SIGMA_PLAN_AH=pn, ANCHOR_R_SOURCE="state")),
        ("joule_shape", "channel 3 alone: uninformed prior, no anchor, beta follows R_BOL(SoC)",
         dict(SOC0_MODE="uninformed", CC_END_ANCHOR=False, SOC_COUPLING="joule_shape")),
        ("split_aging", "Ablation B: phi follows the per-RPT trend", dict(SPLIT_MODE="aging")),
    ]
 
 
def run_ablations(D, therm, luts, cap, phi_trend, out_dir, stride):
    sel = select_events(D["events"], stride)
    print(f"\nAblation suite: {len(sel)} events (stride {stride})")
    rows, long = [], []
    for name, desc, kw in ablation_list():
        if kw.get("SPLIT_MODE") == "aging" and phi_trend is None:
            print(f"  [{name}] skipped: no tier C file for the phi trend")
            continue
        with override(**kw):
            tab, _, info = run_config(D, therm, luts, cap, phi_trend, sel, label=f"[{name}] ")
            s = summarise(tab)
        rows.append(dict(config=name, description=desc, overrides=json.dumps(kw), runtime_s=round(info["runtime_s"], 1), **s))
        tab.insert(0, "config", name)
        long.append(tab)
        print(f"  [{name:<22}] V RMSE online {s['rmse_online_mV']:5.1f} | post {s['rmse_post_mV']:5.1f} | floor "
              f"{s['rmse_floor_mV']:5.1f} mV ({s['rmse_post_pct']:.2f} % NRMSE post) ; SoC0 err rms prior {s['soc0_prior_err_rms_pct']:4.2f} % -> est "
              f"{s['soc0_hat_err_rms_pct']:4.2f} % (mean {s['soc0_hat_err_mean_pct']:+.2f}) ; |CV dt| online "
              f"{s['cv_dt_online_mean_abs_s']:4.0f} s  ({info['runtime_s']:.0f} s)", flush=True)
    abl = pd.DataFrame(rows)
    abl.to_csv(os.path.join(out_dir, f"{D['name']}_step3_ablations.csv"), index=False)
    pd.concat(long, ignore_index=True).to_csv(os.path.join(out_dir, f"{D['name']}_step3_ablation_events.csv"), index=False)
    if MAKE_PLOTS:
        plot_ablations(abl, os.path.join(out_dir, f"{D['name']}_step3_ablations.png"), D["name"], len(sel))
    return abl
 
 
# 10. Self-tests 
 
def soc0_cost_profile(T_cc, T_amb, alpha, dt, sigma_T, dsoc, luts, s0_grid, coupling):
# Self-test 5: full-information cost (all samples, no arrival cost, as Eq. 9) with SoC0 FIXED, for each SoC0
# With the SoC path fixed, beta+ = rho_i beta is linear, so each point is an exact linear least-squares fit
    n = len(T_cc)
    e = float(np.exp(-alpha * dt))
    c = (1.0 - e) / alpha
    J = []
    for s0 in s0_grid:
        soc = s0 + dsoc * np.arange(n)
        rho = luts.rbol(soc[1:]) / luts.rbol(soc[:-1]) if coupling == "joule_shape" else np.ones(n - 1)
        A = np.zeros((n + 2 * (n - 1), 2 * n))
        rhs = np.zeros(A.shape[0])
        r = 0
        for i in range(n): # T measurements
            A[r, i] = 1.0 / sigma_T
            rhs[r] = T_cc[i] / sigma_T
            r += 1
        for i in range(n - 1): # T dynamics
            A[r, i + 1], A[r, i], A[r, n + i] = 1.0 / Q_T_K, -e / Q_T_K, -c / Q_T_K
            rhs[r] = (1.0 - e) * T_amb / Q_T_K
            r += 1
        for i in range(n - 1): # beta dynamics
            A[r, n + i + 1], A[r, n + i] = 1.0 / Q_BETA_KPS, -rho[i] / Q_BETA_KPS
            r += 1
        z = np.linalg.lstsq(A, rhs, rcond=None)[0]
        J.append(float(np.sum((A @ z - rhs) ** 2)))
    return np.array(J)
 
 
def self_tests(D, therm, luts, cap, out_dir):
    evs = D["events"]
    pick = sorted(set([0, 1, len(evs) // 4, len(evs) // 2, (3 * len(evs)) // 4, len(evs) - 1]))
    res, ok_all = {}, True
    cfg1, dt = D["cfg1"], therm["dt"]
    prepared = []
    t_last = np.nan
    for i in pick:
        P = prepare_event(evs[i], D, therm, luts, cap, 1.0, t_last)
        prepared.append((evs[i], P))
 
    # Test 1: regression to Step 1 ([T, beta] identical with SOC_COUPLING = "none") 
    print("\nTest 1: [T, beta] vs Step 1's ThermalMHE on the same inputs")
    st1 = s1.ThermalMHE(therm["alpha"], dt, therm["beta_f"], therm["sigma_T"], cfg1)
    ref = {ev["event"]: st1.run_event(P["T_cc"], P["t_amb"]) for ev, P in prepared}
    t1 = []
    for anchor in (False, True):
        with override(CC_END_ANCHOR=anchor, SOC_COUPLING="none", ANCHOR_R_SOURCE="trailing"):
            est = ExtendedMHE(luts, therm["alpha"], dt, therm["beta_f"], therm["sigma_T"], sigma_soc0_for_mode(),
                              therm["beta_0"], therm["rbol_ref"])
            for ev, P in prepared:
                out = est.run_event(P["T_cc"], P["t_amb"], P["soc0_prior"], P["dsoc"], P["anc"])
                T1, b1 = ref[ev["event"]]
                bk3, bk1 = s1.beta_k_from_trace(out["beta"], TAIL_FRAC), s1.beta_k_from_trace(b1, TAIL_FRAC)
                t1.append(dict(anchor=anchor, event=ev["event"], max_dT_K=float(np.max(np.abs(out["T_hat"] - T1))),
                               max_dbeta=float(np.max(np.abs(out["beta"] - b1))), beta_k_rel=abs(bk3 / bk1 - 1)))
    t1 = pd.DataFrame(t1)
    print(t1.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
    p1 = bool((t1["max_dT_K"] < 1e-4).all() and (t1["max_dbeta"] < 1e-6).all() and (t1["beta_k_rel"] < 1e-4).all())
    print(f"  -> {'PASS' if p1 else 'FAIL'} (tolerance |dT| < 1e-4 K, |dbeta| < 1e-6 K/s, beta_k rel < 1e-4)")
    res["test1_regression"] = dict(passed=p1, rows=t1.to_dict("records"))
    ok_all &= p1
 
    # Test 2: scalar cross-check (block-diagonal defaults) 
    print("\nTest 2: MHE SoC0 vs the 1-D solution (prior + one anchor equation)")
    t2 = []
    for mode, lin in (("duty_cycle", False), ("uninformed", False), ("uninformed", True)):
        with override(SOC0_MODE=mode, LINEARIZE_ANCHOR=lin, CC_END_ANCHOR=True, ANCHOR_R_SOURCE="trailing",
                      SOC_COUPLING="none"):
            sig = sigma_soc0_for_mode()
            est = ExtendedMHE(luts, therm["alpha"], dt, therm["beta_f"], therm["sigma_T"], sig, therm["beta_0"],
                              therm["rbol_ref"])
            for ev, P in prepared:
                prior = soc0_prior_for(ev, P["q_k"], P["soc0_true"])
                anc = dict(P["anc"])
                anc["s_lin"] = prior + (P["n"] - 1) * P["dsoc"] + anc["ds"]
                anc["v_lin"], anc["g_lin"] = luts.vq_num(anc["s_lin"], anc["r"])
                out = est.run_event(P["T_cc"], P["t_amb"], prior, P["dsoc"], anc)
                sc = soc0_scalar(luts, prior, sig, P["dsoc"], anc, lin)
                t2.append(dict(prior=mode, linear=lin, event=ev["event"], soc0_prior=prior, soc0_mhe=out["soc0_hat"],
                               soc0_scalar=sc, diff=out["soc0_hat"] - sc, correction=sc - prior))
    t2 = pd.DataFrame(t2)
    print(t2.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    p2 = bool((t2["diff"].abs() < 2e-4).all())
    print(f"  -> {'PASS' if p2 else 'FAIL'} (tolerance 2e-4; the Q_SOC slack absorbs ~1e-3 of a correction)")
    res["test2_scalar"] = dict(passed=p2, rows=t2.to_dict("records"))
    ok_all &= p2
 
    # Test 3: anchor alignment (node <-> sample mapping) 
    print("\nTest 3: anchor moved to an interior sample m with a target implying SoC0 = prior + 0.01")
    ev, P = prepared[len(prepared) // 2]
    m = P["n"] // 2
    with override(CC_END_ANCHOR=True, ANCHOR_R_SOURCE="trailing", SOC_COUPLING="none", LINEARIZE_ANCHOR=False,
                  SOC0_MODE="duty_cycle"):
        sig = sigma_soc0_for_mode()
        est = ExtendedMHE(luts, therm["alpha"], dt, therm["beta_f"], therm["sigma_T"], sig, therm["beta_0"],
                          therm["rbol_ref"])
        prior = P["soc0_prior"]
        anc = dict(P["anc"], i=m, ds=0.0)
        anc["v_target"] = luts.vq_num(prior + 0.01 + m * P["dsoc"], anc["r"])[0]
        out = est.run_event(P["T_cc"], P["t_amb"], prior, P["dsoc"], anc)
        sc = {d: soc0_scalar(luts, prior, sig, P["dsoc"], dict(anc, i=m + d)) for d in (-1, 0, 1)}
    implied = out["soc_filt"][m] - m * P["dsoc"]
    before = out["soc_filt"][m - 1] - (prior + (m - 1) * P["dsoc"])
    gap = min(abs(sc[-1] - sc[0]), abs(sc[1] - sc[0]))
    p3 = bool(abs(implied - sc[0]) < 2e-4 and abs(before) < 1e-6 and gap > 10 * 2e-4)
    print(f"  event {ev['event']}, m = {m}: SoC0 implied at step m {implied:.5f}, scalar(m) {sc[0]:.5f}, "
          f"scalar(m-1) {sc[-1]:.5f}, scalar(m+1) {sc[1]:.5f}; step m-1 untouched by {before:+.1e}")
    print(f"  -> {'PASS' if p3 else 'FAIL'} (|implied - scalar(m)| < 2e-4, off-by-one would be off by >= {gap:.4f})")
    res["test3_alignment"] = dict(passed=p3, event=int(ev["event"]), m=int(m), implied=implied, scalar_m=sc[0],
                                  scalar_m_minus_1=sc[-1], scalar_m_plus_1=sc[1], before=before)
    ok_all &= p3
 
    print("\nTest 4: synthetic end-to-end -- not run (make_synthetic_vah.py does not write thermal traces here)")
 
    # Test 5: observability profile for channel 3 
    print("\nTest 5: full-information cost vs SoC0 with beta following R_BOL(SoC) (no prior, no anchor)")
    idx = np.unique(np.linspace(0, len(evs) - 1, PROFILE_N_EVENTS).astype(int))
    prof, t5 = [], []
    t_last = np.nan
    for i in idx:
        ev = evs[i]
        P = prepare_event(ev, D, therm, luts, cap, 1.0, t_last)
        grid = P["soc0_true"] + np.arange(-0.30, 0.2001, 0.005)
        grid = grid[(grid > 0.0) & (grid + P["dsoc"] * P["n"] < 1.0)]
        J = soc0_cost_profile(P["T_cc"], P["t_amb"], therm["alpha"], dt, therm["sigma_T"], P["dsoc"], luts, grid,
                              "joule_shape")
        J0 = soc0_cost_profile(P["T_cc"], P["t_amb"], therm["alpha"], dt, therm["sigma_T"], P["dsoc"], luts,
                               grid[[0, -1]], "none")
        j_true = float(np.interp(P["soc0_true"], grid, J))
        jm = float(J.min())
        d5 = float(max(np.interp(P["soc0_true"] - 0.05, grid, J), np.interp(P["soc0_true"] + 0.05, grid, J)) - jm)
        im = int(np.argmin(J))
        t5.append(dict(event=ev["event"], soc0_true=P["soc0_true"], argmin_minus_true=float(grid[im] - P["soc0_true"]),
                       argmin_at_edge=bool(im in (0, len(grid) - 1)), J_true_minus_min=j_true - jm, dJ_at_pm5pct=d5,
                       J_none_flat=float(abs(J0[1] - J0[0]))))
        prof += [dict(event=ev["event"], soc0=float(g), J=float(j)) for g, j in zip(grid, J)]
    t5 = pd.DataFrame(t5)
    print(t5.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    flat = bool((t5["dJ_at_pm5pct"] < 4.0).mean() > 0.5)
    biased = bool((t5["argmin_minus_true"].abs() > 0.03).mean() > 0.5)
    if flat:
        verdict = "weakly observable: the cost changes < 4 within +/-5 % SoC0 for most events"
    elif biased:
        verdict = ("informative but BIASED: the minimum sits > 3 % SoC from the truth for most events, so channel 3 "
                   "fits something other than R_BOL(SoC) (entropic heat?). Keep SOC_COUPLING = 'none'.")
    else:
        verdict = "informative and near the truth for most events"
    print(f"  -> {verdict}")
    res["test5_profile"] = dict(rows=t5.to_dict("records"), mostly_flat=flat, mostly_biased=biased, verdict=verdict)
    pd.DataFrame(prof).to_csv(os.path.join(out_dir, f"{D['name']}_step3_soc0_profile.csv"), index=False)
 
    res["all_passed"] = bool(ok_all)
    with open(os.path.join(out_dir, f"{D['name']}_step3_selftest.json"), "w") as fh:
        json.dump(s2.to_jsonable(res), fh, indent=1)
    print(f"\nSelf-tests 1-3: {'ALL PASS' if ok_all else 'FAILURES above'}")
    return res
 
 
# 11. Plots
 
def _style(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel, color=C_INK2)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_INK2)
    if title:
        ax.set_title(title, fontsize=10, color=C_INK, loc="left")
    ax.grid(True, color=C_GRID, lw=0.8)
    ax.tick_params(colors=C_INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbdbd")
 
 
def _protocol_band(ax):
    lo, hi = PROTOCOL_CHANGE_EVENTS
    ax.axvspan(lo, hi, color=C_MEAS, alpha=0.08, lw=0)
 
 
def plot_soc0(tab, path, title, stride):
    h = max(1, int(round(SMOOTH_HALF_WINDOW / stride)))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ev = tab["event"]
    a1.plot(ev, tab["soc0_true"], color=C_INK2, lw=1.6, label="true (coulomb count, Q from RPTs) [dev]")
    a1.plot(ev, tab["soc0_prior"], color=C_ONLINE, lw=1.2, ls="--", label="prior")
    a1.plot(ev, tab["soc0_hat"], color=C_POST, lw=1.2, label="MHE estimate (post)")
    _protocol_band(a1)
    _style(a1, None, "SoC0", title)
    a1.legend(frameon=False, fontsize=8)
    ep, eh = 100 * (tab["soc0_prior"] - tab["soc0_true"]), 100 * (tab["soc0_hat"] - tab["soc0_true"])
    a2.plot(ev, ep, ".", ms=2.5, color=C_ONLINE, alpha=0.35)
    a2.plot(ev, eh, ".", ms=2.5, color=C_POST, alpha=0.35)
    a2.plot(ev, centred_ma(ep, h), color=C_ONLINE, lw=1.6, ls="--", label="prior - true")
    a2.plot(ev, centred_ma(eh, h), color=C_POST, lw=1.6, label="estimate - true")
    a2.axhline(0, color=C_INK2, lw=0.8)
    _protocol_band(a2)
    _style(a2, "1C charge event (chronological, Step 1 numbering)", "SoC0 error [% SoC]",
           f"lines: +/-{SMOOTH_HALF_WINDOW}-cycle average; shaded: protocol change")
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def plot_rmse(tab, path, title, stride):
    h = max(1, int(round(SMOOTH_HALF_WINDOW / stride)))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ev = tab["event"]
    ax.plot(ev, tab["rmse_post_mV"], ".", ms=2.5, color=C_POST, alpha=0.25)
    for col, c, ls, lab in (("rmse_online_mV", C_ONLINE, "-", "online (causal, prior SoC0, trailing r)"),
                            ("rmse_post_mV", C_POST, "-", "post (MHE SoC0, this charge's r)"),
                            ("rmse_soc_only_mV", C_SOC, "--", "est. SoC0 + oracle r [dev]"),
                            ("rmse_r_only_mV", C_R, "-.", "true SoC0 + est. r [dev]"),
                            ("rmse_floor_mV", C_MEAS, ":", "floor: true SoC0 + oracle r [dev]")):
        ax.plot(ev, centred_ma(tab[col], h), color=c, ls=ls, lw=1.7, label=lab)
    _protocol_band(ax)
    _style(ax, "1C charge event (chronological, Step 1 numbering)", "V RMSE over the CC [mV]", title)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def plot_traces(traces, path):
    if not traces:
        return
    fig, axs = plt.subplots(2, len(traces), figsize=(5.2 * len(traces), 6.2), squeeze=False, sharex="col")
    for j, (ev, tr) in enumerate(sorted(traces.items())):
        lab = "BOL" if j == 0 else "EOL"
        t = tr["t"] / 60.0
        a, r = axs[0, j], axs[1, j]
        a.plot(t, tr["V"], "o", ms=3, color=C_MEAS, label="measured [dev]")
        a.plot(t, tr["V_floor"], ":", color=C_INK2, lw=1.5, label="floor (true SoC0, oracle r)")
        a.plot(t, tr["V_on"], color=C_ONLINE, lw=1.6, label="online V_hat")
        a.plot(t, tr["V_post"], color=C_POST, lw=1.6, label="post V_hat")
        _style(a, None, "V [V]", f"{lab}: event {ev}")
        a.legend(frameon=False, fontsize=7)
        for k, c, ls in (("V_on", C_ONLINE, "-"), ("V_post", C_POST, "-"), ("V_floor", C_INK2, ":")):
            r.plot(t, 1e3 * (tr[k] - tr["V"]), color=c, ls=ls, lw=1.5)
        r.axhline(0, color=C_INK2, lw=0.8)
        _style(r, "minutes into CC", "V_hat - measured [mV]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def plot_ablations(abl, path, name, n_events):
    abl = abl.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(abl))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.42 * len(abl) + 1.6), sharey=True)
    a1.plot(abl["rmse_floor_mV"], y, "|", ms=12, mew=2, color=C_MEAS, label="floor (true SoC0, oracle r)")
    a1.plot(abl["rmse_online_mV"], y, "o", ms=6, color=C_ONLINE, label="online")
    a1.plot(abl["rmse_post_mV"], y, "o", ms=6, color=C_POST, label="post")
    for yi, v, w in zip(y, abl["rmse_post_mV"], abl["rmse_online_mV"]):
        a1.annotate(f"post {v:.1f}", (max(v, w), yi), xytext=(8, -3), textcoords="offset points", fontsize=7,
                    color=C_INK2)
    a1.set_yticks(y)
    a1.set_yticklabels(abl["config"], fontsize=8)
    _style(a1, "mean V RMSE over the CC [mV]", None, f"{name}: ablations ({n_events} events)")
    a1.set_xlim(left=0)
    a1.legend(frameon=False, fontsize=7, loc="lower right")
    a2.plot(abl["soc0_prior_err_rms_pct"], y, "o", ms=6, mfc="white", mec=C_ONLINE, mew=1.5, label="prior")
    a2.plot(abl["soc0_hat_err_rms_pct"], y, "o", ms=6, color=C_POST, label="MHE estimate")
    _style(a2, "SoC0 error, RMS over events [% SoC]", None, "SoC0 vs truth [dev]")
    a2.set_xlim(left=0)
    a2.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
# 12. Main
 
def main(argv=None):
    global SOC0_MODE, CC_END_ANCHOR, ANCHOR_R_SOURCE, SOC_COUPLING, LINEARIZE_ANCHOR, SIGMA_PLAN_AH, \
        R_SCALE_SOURCE, SPLIT_MODE, FADE_MODE, MAKE_PLOTS, SOC_FRAME
    ap = argparse.ArgumentParser(description="Step 3: extended MHE [T, beta, SoC] with V(t) as a derived output.")
    ap.add_argument("--csv", required=True, help="cell time-series CSV (e.g. VAH01.csv)")
    ap.add_argument("--handoff", default=None, help="Step 2 hand-off JSON (default step2_results/<cell>_ecm_bol.json)")
    ap.add_argument("--fade-csv", default=None, help="training-cell <cell>_rpt_capacity.csv for Q_fade (optional)")
    ap.add_argument("--tierc", default=None, help="Step 2 <cell>_tierC_rpt_steps.csv for Ablation B (optional)")
    ap.add_argument("--out", default="results_step3")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--soc0-mode", choices=["duty_cycle", "uninformed", "oracle"], default=SOC0_MODE)
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--anchor-r", choices=["trailing", "state", "oracle"], default=ANCHOR_R_SOURCE)
    ap.add_argument("--coupling", choices=["none", "joule_shape"], default=SOC_COUPLING)
    ap.add_argument("--linearize", action="store_true", help="linearised anchor (Ablation 3)")
    ap.add_argument("--sigma-plan-ah", type=float, default=SIGMA_PLAN_AH)
    ap.add_argument("--r-scale", choices=["event", "trailing", "oracle"], default=R_SCALE_SOURCE)
    ap.add_argument("--split", choices=["fixed", "aging"], default=SPLIT_MODE)
    ap.add_argument("--fade-mode", choices=["fit", "causal", "rpt_interp"], default=FADE_MODE)
    ap.add_argument("--soc-frame", choices=["auto", "forward", "backward"], default=SOC_FRAME,
                    help="mission-Ah frame for truth and prior (auto = the hand-off's frame)")
    ap.add_argument("--ablations", action="store_true", help="also run the ablation matrix")
    ap.add_argument("--ablations-only", action="store_true", help="run only the ablation matrix (no main run)")
    ap.add_argument("--ablation-stride", type=int, default=ABLATION_STRIDE)
    ap.add_argument("--self-test", action="store_true", help="run self-tests 1, 2, 3, 5 and stop")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    SOC0_MODE, CC_END_ANCHOR, ANCHOR_R_SOURCE = a.soc0_mode, not a.no_anchor, a.anchor_r
    SOC_COUPLING, LINEARIZE_ANCHOR, SIGMA_PLAN_AH = a.coupling, a.linearize, a.sigma_plan_ah
    R_SCALE_SOURCE, SPLIT_MODE, FADE_MODE = a.r_scale, a.split, a.fade_mode
    MAKE_PLOTS = MAKE_PLOTS and not a.no_plots
    t_start = time.time()
    os.makedirs(a.out, exist_ok=True)
 
    print(f"loading {a.csv} ...", flush=True)
    D = load_data(a.csv)
    name = D["name"]
    hp = a.handoff or os.path.join("step2_results", f"{name}_ecm_bol.json")
    if not os.path.exists(hp):
        sys.exit(f"Step 2 hand-off not found: {hp}. Run step2_ecm_calibration.py first (or pass --handoff).")
    with open(hp) as fh:
        handoff = json.load(fh)
    ho_frame = handoff.get("provenance", {}).get("soc0_ref", "forward")
    SOC_FRAME = ho_frame if a.soc_frame == "auto" else a.soc_frame
    if SOC_FRAME != ho_frame:
        print(f"  WARNING: SoC frame '{SOC_FRAME}' differs from the frame the hand-off was calibrated in "
              f"('{ho_frame}'); the frozen ECM will be offset by the frame difference (~2.5 % SoC at BOL on VAH01)")
    tierc = a.tierc or os.path.join(os.path.dirname(hp), f"{name}_tierC_rpt_steps.csv")
    phi_trend, phi_coef = load_phi_trend(tierc)
    luts = Luts(handoff)
    D["ecm"] = load_ecm(handoff)
    cap = Capacity(D["blocks"], a.fade_csv)
    imb = np.array([e["ah_out"] - e["ah_back"] for e in D["events"]])
    print(f"  SoC frame '{SOC_FRAME}' (hand-off: '{ho_frame}'); Ah out - Ah back per charge: first 10 % "
          f"{imb[:len(imb) // 10].mean():+.3f} Ah, last 10 % {imb[-(len(imb) // 10):].mean():+.3f} Ah")
    print(f"  {len(D['events'])} 1C CC events, {len(D['blocks'])} RPTs; Q_BOL {cap.q_bol:.4f} Ah; fade curve from "
          f"{cap.source}{' -- in-sample, use --fade-csv from a training cell for an honest test' if cap.in_sample else ''}")
    print(f"  LUTs ({luts.kind}): OCV monotone {luts.ocv_monotone}, max deviation from the hand-off grid "
          f"{luts.max_dev_ocv_mV:.3f} mV / {luts.max_dev_r_mOhm:.3f} mOhm; R_cc measured on SoC {luts.rcc_range}")
    therm = calibrate_thermal(D, luts, cap)
    print(f"  alpha {therm['alpha']:.3e} 1/s, sigma_T {therm['sigma_T']:.3f} K, dt {therm['dt']:.0f} s, beta_0 "
          f"{therm['beta_0']:.3e} K/s, beta_f {therm['beta_f']:.3e} K/s, R_BOL ref {1e3 * therm['rbol_ref']:.1f} mOhm")
 
    if a.self_test:
        self_tests(D, therm, luts, cap, a.out)
        return 0
    if a.ablations_only:
        run_ablations(D, therm, luts, cap, phi_trend, a.out, a.ablation_stride)
        print(f"Done in {time.time() - t_start:.0f} s. Outputs in {a.out}")
        return 0
 
    sel = select_events(D["events"], a.stride, a.max_events)
    print(f"\nMain run: {len(sel)} events | SoC0 {SOC0_MODE} (sigma {sigma_soc0_for_mode():.3f}), anchor "
          f"{'on (r ' + ANCHOR_R_SOURCE + ', ' + ('linear' if LINEARIZE_ANCHOR else 'nonlinear') + ')' if CC_END_ANCHOR else 'off'}, "
          f"coupling {SOC_COUPLING}, post r {R_SCALE_SOURCE}, split {SPLIT_MODE}, fade {FADE_MODE}", flush=True)
    tab, traces, info = run_config(D, therm, luts, cap, phi_trend, sel, keep_traces=(sel[0]["event"], sel[-1]["event"]))
    s = summarise(tab)
    tab.to_csv(os.path.join(a.out, f"{name}_step3_events.csv"), index=False)
    fade_err = 100 * rmse(tab["Q_k"] / tab["Q_rpt"] - 1)
 
    print(f"\n  V RMSE (mean over events): online {s['rmse_online_mV']:.1f} mV | post {s['rmse_post_mV']:.1f} mV | "
          f"floor {s['rmse_floor_mV']:.1f} mV;  early charge: online {s['rmse_online_early_mV']:.1f}, post "
          f"{s['rmse_post_early_mV']:.1f} mV")
    print(f"  NRMSE (% of {V_WINDOW_LO:.1f}-{V_WINDOW_HI:.1f} V window): online {s['rmse_online_pct']:.2f} % | post "
          f"{s['rmse_post_pct']:.2f} % | floor {s['rmse_floor_pct']:.2f} %;  first / last 10 % post "
          f"{s['first10_rmse_post_pct']:.2f} -> {s['last10_rmse_post_pct']:.2f} %")
    print(f"  first / last 10 % of life: post {s['first10_rmse_post_mV']:.1f} -> {s['last10_rmse_post_mV']:.1f} mV, "
          f"floor {s['first10_rmse_floor_mV']:.1f} -> {s['last10_rmse_floor_mV']:.1f} mV, r_oracle "
          f"{s['first10_r_oracle']:.2f} -> {s['last10_r_oracle']:.2f}, r_event {s['first10_r_event']:.2f} -> "
          f"{s['last10_r_event']:.2f}")
    print(f"  SoC0 error: prior rms {s['soc0_prior_err_rms_pct']:.2f} % (mean {s['soc0_prior_err_mean_pct']:+.2f}) -> "
          f"estimate rms {s['soc0_hat_err_rms_pct']:.2f} % (mean {s['soc0_hat_err_mean_pct']:+.2f}); estimate bias "
          f"first / last 10 %: {s['first10_soc0_hat_err_mean_pct']:+.2f} / {s['last10_soc0_hat_err_mean_pct']:+.2f} %")
    print(f"  H2 attribution (added RMSE over the floor): SoC part {s['h2_soc_part_mV']:.1f} mV vs resistance part "
          f"{s['h2_r_part_mV']:.1f} mV;  corr(SoC0 error, r error) {s['conflation_corr_soc0err_vs_rerr']:+.2f}")
    print(f"  online end-of-CC {s['end_err_online_mean_mV']:+.1f} mV, CC->CV timing {s['cv_dt_online_mean_s']:+.0f} s "
          f"(|.| {s['cv_dt_online_mean_abs_s']:.0f} s, {s['cv_online_not_reached']} not reached)"
          + (f";  post: {s['end_err_post_mean_mV']:+.1f} mV, |CV dt| {s['cv_dt_post_mean_abs_s']:.0f} s "
             f"(circular: forced by the anchor)" if CC_END_ANCHOR else ""))
    print(f"  checks: scalar max |diff| {s['scalar_check_max_abs']:.1e}, anchor SoC outside the measured R range "
          f"{100 * s['anchor_extrap_frac']:.0f} %, solver failures {s['solver_fail_steps']}, fade-curve Q error "
          f"rms {fade_err:.2f} %, {info['ms_per_step']:.1f} ms/step")
 
    if MAKE_PLOTS:
        ttl = f"{name}: SoC0 {SOC0_MODE}, anchor {'on' if CC_END_ANCHOR else 'off'}, coupling {SOC_COUPLING}"
        plot_soc0(tab, os.path.join(a.out, f"{name}_step3_soc0.png"), ttl, a.stride)
        plot_rmse(tab, os.path.join(a.out, f"{name}_step3_rmse.png"), ttl, a.stride)
        plot_traces(traces, os.path.join(a.out, f"{name}_step3_traces.png"))
 
    abl = None
    if a.ablations:
        abl = run_ablations(D, therm, luts, cap, phi_trend, a.out, a.ablation_stride)
 
    summary = dict(cell=name, runtime_s=round(time.time() - t_start, 1), handoff=os.path.abspath(hp),
                   n_events_total=len(D["events"]), n_events_run=len(sel), stride=a.stride, soc_frame=SOC_FRAME,
                   handoff_soc_frame=ho_frame, ah_out_minus_back_mean=float(imb.mean()), thermal=therm,
                   luts=dict(kind=luts.kind, ocv_monotone=luts.ocv_monotone, max_dev_ocv_mV=luts.max_dev_ocv_mV,
                             max_dev_r_mOhm=luts.max_dev_r_mOhm, rcc_soc_range=list(luts.rcc_range)),
                   capacity=dict(mode=FADE_MODE, source=cap.source, in_sample=cap.in_sample, q_bol_Ah=cap.q_bol,
                                 fade_coef_normalised=cap.coef.tolist(), q_fade_err_rms_pct=fade_err),
                   phi_trend_coef=phi_coef, ms_per_step=info["ms_per_step"], metrics=s,
                   notes=["post end-of-CC error and CV timing are circular when CC_END_ANCHOR is on; use the online ones",
                          "duty_cycle prior uses the dataset's mission Ah in SOC_FRAME as the plan (SIGMA_PLAN_AH adds "
                          "plan noise)",
                          "soc0_true, r_oracle, floor/r_only/soc_only use measured V/I: [DEV-ONLY]"],
                   ablations=abl.to_dict("records") if abl is not None else None, settings=settings_dict())
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(s2.to_jsonable(summary), fh, indent=1)
    print(f"Done in {time.time() - t_start:.0f} s. Outputs in {a.out}")
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())