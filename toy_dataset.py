import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- Device ----------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ---------- Toy environment hyperparams ----------
DT = 1.0
PROCESS_NOISE_STD = 0.02   # epsilon_t ~ N(0, sigma^2)
K_GENTLE = 0.5             # gentle correction gain
K_STRONG = 1.5             # strong correction when far
STRONG_THRESHOLD = 1.0     # |x| > threshold -> strong correction
U_MAX = 0.5                # clip steering

TRAJ_LEN = 50              # length of each rollout
NUM_TRAJ = 2000            # number of expert trajectories to generate
CONTEXT_LEN = 16           # how many past states to feed into the transformer

TRAIN_STATE_LIMIT = 1.2    # only keep training examples where |x| <= this
TEST_PERTURB_STD = 0.3     # for test, start a bit more perturbed

ON_MANIFOLD_ONLY = False # whether to only keep training examples where |x| <= TRAIN_STATE_LIMIT

rng = np.random.default_rng(0)


def expert_policy(x):
    """
    Simple "human" controller:
    - near center: gentle PD-ish correction
    - far away: strong correction
    """
    if abs(x) < STRONG_THRESHOLD:
        u = -K_GENTLE * x
    else:
        u = -K_STRONG * x
    # get cubic root of u
    u = np.cbrt(u) + 0.1

    # Clip steering (humans are bounded)
    u = np.clip(u, -U_MAX, U_MAX)
    return u


def simulate_expert_trajectory(x0=0.0):
    xs = [x0]
    us = []
    x = x0
    for t in range(TRAJ_LEN):
        u = expert_policy(x)
        us.append(u)
        noise = rng.normal(0.0, PROCESS_NOISE_STD)
        x = x + (u-0.1)**3 * DT + noise
        xs.append(x)
    return np.array(xs), np.array(us)  # xs has length TRAJ_LEN+1, us has TRAJ_LEN


def generate_dataset(num_traj=NUM_TRAJ):
    """
    Generate (states, actions) pairs from expert rollouts, then
    slice into context windows for training.
    """
    contexts = []
    targets = []

    for _ in range(num_traj):
        x0 = rng.normal(0.0, 0.2)  # expert typically starts near center
        xs, us = simulate_expert_trajectory(x0)

        # Sliding windows: predict u_{t-1} from states [t-C, ..., t-1]
        for t in range(CONTEXT_LEN, TRAJ_LEN):
            # we use previous CONTEXT_LEN states as input
            window_states = xs[t-CONTEXT_LEN:t]  # length = CONTEXT_LEN
            target_u = us[t-1]

            # Filter to mimic "on-manifold only" training data
            if ON_MANIFOLD_ONLY:
                if abs(window_states[-1]) <= TRAIN_STATE_LIMIT:
                    contexts.append(window_states.astype(np.float32))
                    targets.append(np.float32(target_u))
            else:
                contexts.append(window_states.astype(np.float32))
                targets.append(np.float32(target_u))

    contexts = np.stack(contexts, axis=0)  # (N, CONTEXT_LEN)
    targets = np.stack(targets, axis=0)    # (N,)

    print("Generated dataset:", contexts.shape, targets.shape)
    return contexts, targets


class ImitationDataset(Dataset):
    def __init__(self, contexts, targets):
        self.contexts = torch.from_numpy(contexts)  # (N, T)
        self.targets = torch.from_numpy(targets)    # (N,)

    def __len__(self):
        return self.contexts.shape[0]

    def __getitem__(self, idx):
        x_seq = self.contexts[idx].unsqueeze(-1)  # (T, 1)
        u = self.targets[idx]                     # scalar
        return x_seq, u
