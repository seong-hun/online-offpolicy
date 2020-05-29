import numpy as np
import os

# Simulation setup
TIME_STEP = 1e-2
FINAL_TIME = 50
INITIAL_STATE = np.vstack((-1, -1))

# Actor Critic
ETA = 1e2
ETA_PI = 1e2
TAUF = 1e-2
K = 1e2
R = 1
Q = np.eye(2)
