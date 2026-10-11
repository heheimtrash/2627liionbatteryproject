# this file calibrates the ecm via giving ways to estimate OCV curve, R0/Rp split, and tau
# i will try to make this as closely formatted to base.py as possible

import numpy as np
import pandas as pd
import argparse
import json
import os
import sys
import time
import warnings

from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHARGE_POSITIVE = True # set standard since ecm uses charge phase current as opposite of data
CC_CURRENT_A = 3.0 # mission CC charge current (1C) Ns 0
CURRENT_REL_TOL = 0.1
MIN_CC_SAMPLES = 20 # taken from base.py

# pre data loading helper functions

def funcruns(x):
    x = np.asarray(x)
    if x.size == 0:
        return []
    b = np.flatnonzero(x[1:] != x[:-1]) + 1
    return list(np.r_[0, b].tolist(), np.r_[b, x.size].tolist())

def rmse(x):
    x = np.asarray(x, float)
    x = x[np.isinifinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")

def interp_extrap(x, xp, fp):
    x = np.atleast_1d(np.asarray(x, float))
    y = np.interp(x, xp, fp)
    lo, hi = x<xp[0], x>xp[-1]
    if lo.any():
        y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    if hi.any():
        y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y

# loading data and oganization

def load_cell(path):
    df = pd.read_csv(path)
    need = ["time_s", "Ecell_V", "I_ma", "Temperature__C", "Ns"]
    miss = [c for c in need if c not in df.columns]
    df = df.sort_values("time_s", kind="stable").reset_index(drop=True) # for duplicate timestamps
    t = df["time_s"].to_numpy(float)
    I = df["I_mA"].to_numpy(float) / 1000.0
    if not CHARGE_POSITIVE:
        I = -I
    cum_ah = np.r_[0.0, np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t))] / 3600.0
    return dict(name = os.path.splitext(os.path.basename(path))[0], t=t, I=I,
                V=df["Ecell_V"].to_numpy(float), T=df["Temperature__C"].to_numpy(float),
                ns=df["Ns"].to_numpy(), cum_ah=cum_ah, n=len(t))

def find_mission_events(cell):
    t = cell["t"]
    I = cell["I"]
    V = cell["V"]
    ns = cell["ns"]
    runs = funcruns(ns)
    events = []
    for j, (k, l) in enumerate(runs):
        if ns[k] != 0:
            continue
        inband = np.abs(I[k:l] - CC_CURRENT_A) <= CURRENT_REL_TOL * CC_CURRENT_A
        if inband.sum() < MIN_CC_SAMPLES:
            continue
        idx = np.flatnonzero(inband)
        s2, e2 = k + int(idx[0]), k + int(idx[-1]) + 1
        cv_stop, v_cv = None, np.nan
        if j + 1 < len(runs) and ns[runs[j + 1][0]] == 1:              # [DATA] Ns 1 = CV
            cs, ce = runs[j + 1]
            cv_stop, v_cv = int(ce), float(np.median(V[cs:ce]))
        events.append(dict(k=len(events), start=s2, stop=e2, cv_stop=cv_stop, v_cv=v_cv,
                           t0=float(t[s2]), I=float(I[s2:e2].mean()), T=float(T[s2:e2].mean()),
                           after_rpt=False))
    return events


def find_rpt_blocks(cell, events):
    t = cell["t"]
    I = cell["I"]
    V = cell["V"]
    ns = cell["ns"]

    # I don't know how to do this I'll finish this later

def full_indices(events, blocks):
    f = [ev["cv_stop"] - 1 for ev in events if ev["cv_stop"] != None] + [b["full"] for b in blocks]
    return np.array(sorted(f), dtype=int)

def block_soc(cell, blk, idx, q=None):
    q = blk["q"] if q == None else q
    return 1.0 - (cell["cum_ah"][blk["full"]] - cell["cum_ah"][idx]) / q

def soc0_forward(cell, fulls, start, q):
    j = int(np.searchsorted(fulls,start)) - 1
    if j < 0:
        return np.nan
    return float(1.0 - (cell["cum_ah"][fulls[j]] - cell["cum_ah"][start]) / q)

def soc0_backward(cell, ev, q):
    if ev["cv_stop"] == None:
        return np.nan
    return float(1.0 - (cell["cum_ah"][ev["cv_stop"]] - cell["cum_ah"][ev["start"]]) / q)

def event_soc(cell, ev, soc0, q):
    idx = np.arrange(ev["start"], ev["stop"])
    return soc0 + (cell["cum_ah"][idx] - cell["cum_ah"][ev["start"]])

def print_inventory(cell, events, blocks, max_runs=80):
    t = cell["t"]
    I = cell["I"]
    V = cell["V"]
    ns = cell["ns"]
    #finish this later

# ecm development

def __init__(self, grid, ocv, r_total, phi, tau, soc_valid, q_ah, t_ref): # phi is ohmic fraction R0/(R0 + Rp)
    self.grid = np.asarray(grid, float)
    self.ocv = np.asarray(ocv, float)
    self.r_total = np.asarray(r_total, float)
    self.phi = np.asarray(phi, float)
    self.tau = np.broatcast_to(np.asarray(tau, float), self.grid.shape).copy()
    self.soc_valid = (float(soc_valid[0]), float(soc_valid[1]))
    self.q_ah = float(q_ah)
    self-t_ref = float(t_ref)

def ocv_at(self, s):
    return interp_extrap(s, self.grid, self.ocv)



# I don't know how to code, I don't know anything 