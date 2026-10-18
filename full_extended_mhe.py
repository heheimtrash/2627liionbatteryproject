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



