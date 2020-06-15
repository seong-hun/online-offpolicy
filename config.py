import numpy as np
import os

# Simulation setup
TIME_STEP = 1e-3
FINAL_TIME = 100
INITIAL_STATE = np.vstack((-1, -1))

# Actor Critic
ETA = 1e1
ETA_PI = 1e2
TAUF = 1e-2
K = 1e2
R = 1
Q = np.eye(2)
WCAST = np.vstack((0.5, 0, 1))
WAAST = np.vstack((0.5, 1))
