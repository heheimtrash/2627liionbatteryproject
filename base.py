# this code is an accurate recreation of Ahuja et al. 2026's methodology, article "Lithium-ion battery State of Health
# estimation based solely on temperature sensing", J. Power Sources 666 239068
# required extra packages: casadi, do-mpc, matplotlib, pandas

import numpy as numpy
import sys
from casadi import *
import os
import time
from dataclasses import dataclass, asdict, field

import do_mpc 
import matplotlib.pyplot as plt
import pandas as pd

# variables explicitly stated by the paper
n_of_horizon = 30
t_rise_max = 60.0
rfinal_over_rinitial = 2.0
n_alpha_events = 5 # first 5 cycles
alpha_paper = 2.2e-3 # thermal dissipation constant
arrival_weight = 1e-6 # eq 9 on paper, no arrival cost
n_betainitial_events = 5
tail_frac = 0.25
smooth_half_window = 25 # +- 25 cycle moving average for graphse

# assumed/ambiguous variables required for mhe in this scenario

