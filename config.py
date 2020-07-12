import numpy as np
import os

# Simulation setup
TIME_STEP = 1e-3
FINAL_TIME = 200
INITIAL_STATE = np.vstack((-1, -1))

# Actor Critic
ETA = 1e1
ETA_PI = 1e2
TAUF = 1e-2
K = 1e2
R = 1
Q = np.eye(2)
WCAST = np.vstack((0.5, 0, 1))
WAAST = np.vstack((-2, -1))
WCINIT = np.vstack((0, 0, 0))
WTINIT = np.vstack((-2, 0))
WAINIT = np.vstack((-2, 0))
