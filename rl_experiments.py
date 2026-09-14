"""Reproducible tabular RL experiments. Run: python rl_experiments.py.

The notebook embeds this source, so it also works as a standalone Colab upload.
Rover task specification: Vizuara AI Labs, RL in Production, Lecture 01.
"""
from pathlib import Path
import argparse
import json
import platform

import gymnasium as gym
from gymnasium import spaces
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

GAMMA = 0.95
GRID = 5
GOAL = (4, 4)
CRATERS = {(2, 2), (1, 3)}
MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
PERP = {0: [3, 1], 1: [0, 2], 2: [1, 3], 3: [2, 0]}

def rc(s):
    return divmod(int(s), GRID)

def idx(r, c):
    return r * GRID + c

# SECTION: environments
class MarsRoverEnv(gym.Env):
    """Twenty-five fully observed cells with perpendicular movement slips."""
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, slip=0.1, render_mode=None):
        super().__init__()
        if not 0 <= slip <= 0.5:
            raise ValueError("slip must lie between 0 and 0.5")
        if render_mode not in (None, "ansi"):
            raise ValueError("Supported render mode: ansi")
        self.slip = slip
        self.render_mode = render_mode
        self.observation_space = spaces.Discrete(25)
        self.action_space = spaces.Discrete(4)
        self.goal = idx(*GOAL)
        self.craters = {idx(*c) for c in CRATERS}
        self.terminals = self.craters | {self.goal}
        self.P = self._build_model()
        self.s = None

    def _move(self, s, a):
        r, c = rc(s)
        dr, dc = MOVES[a]
        return idx(min(max(r + dr, 0), 4), min(max(c + dc, 0), 4))

    def _reward(self, sn):
        return 10.0 if sn == self.goal else -10.0 if sn in self.craters else -1.0

    def _build_model(self):
        P = {s: {a: [] for a in range(4)} for s in range(25)}
        for s in range(25):
            for a in range(4):
                if s in self.terminals:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                outcomes = {}
                for direction, prob in [(a, 1 - 2*self.slip),
                                        (PERP[a][0], self.slip),
                                        (PERP[a][1], self.slip)]:
                    if prob > 0:
                        sn = self._move(s, direction)
                        outcomes[sn] = outcomes.get(sn, 0.0) + prob
                P[s][a] = [(p, sn, self._reward(sn), sn in self.terminals)
                           for sn, p in outcomes.items()]
        return P

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0
        return self.s, {}

    def step(self, action):
        if self.s is None:
            raise RuntimeError("Call reset before step")
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")
        tr = self.P[self.s][int(action)]
        i = self.np_random.choice(len(tr), p=[t[0] for t in tr])
        _, self.s, reward, terminated = tr[i]
        return self.s, reward, terminated, False, {}

    def render(self):
        board = np.full((5, 5), ".", dtype="<U1")
        for r, c in CRATERS:
            board[r, c] = "X"
        board[GOAL] = "G"
        if self.s is not None:
            board[rc(self.s)] = "R"
        return "\n".join(" ".join(row) for row in board)


class WindyDroneEnv(gym.Env):
    """Twelve cells; a 20% eastward gust replaces the intended move."""
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(12)
        self.action_space = spaces.Discrete(4)
        self.goal = 11
        self.terminals = {6, 11}
        self.s = None
        self.P = {s: {} for s in range(12)}
        for s in range(12):
            for a in range(4):
                if s in self.terminals:
                    self.P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                outcomes = {}
                for direction, prob in [(a, 0.8), (1, 0.2)]:
                    r, c = divmod(s, 4)
                    dr, dc = MOVES[direction]
                    sn = min(max(r+dr, 0), 2)*4 + min(max(c+dc, 0), 3)
                    outcomes[sn] = outcomes.get(sn, 0.0) + prob
                self.P[s][a] = [(p, sn, 8.0 if sn == 11 else -6.0 if sn == 6 else -0.5,
                                 sn in self.terminals) for sn, p in outcomes.items()]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0
        return self.s, {}

    def step(self, action):
        if self.s is None:
            raise RuntimeError("Call reset first")
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")
        tr = self.P[self.s][int(action)]
        i = self.np_random.choice(len(tr), p=[t[0] for t in tr])
        _, self.s, reward, term = tr[i]
        return self.s, reward, term, False, {}

# SECTION: dynamic programming
def action_values(env, s, V, gamma):
    return np.array([sum(p * (r + gamma * V[sn] * (not term))
                         for p, sn, r, term in env.P[s][a])
                     for a in range(env.action_space.n)])


def value_iteration(env, gamma=0.95, theta=1e-6):
    V = np.zeros(env.observation_space.n)
    for _ in range(100_000):
        delta = 0.0
        for s in range(len(V)):
            if s in env.terminals:
                continue
            new_value = np.max(action_values(env, s, V, gamma))
            delta = max(delta, abs(new_value - V[s]))
            V[s] = new_value
        if delta < theta:
            break
    else:
        raise RuntimeError("Value iteration failed to converge")
    policy = np.full(len(V), -1, dtype=int)
    for s in range(len(V)):
        if s not in env.terminals:
            policy[s] = np.argmax(action_values(env, s, V, gamma))
    return V, policy


def policy_grid(policy, shape, terminals, goal):
    symbols = ["^", ">", "v", "<"]
    return np.array(["G" if s == goal else "X" if s in terminals else symbols[a]
                     for s, a in enumerate(policy)]).reshape(shape)


def exact_policy_value(env, policy, gamma=0.95):
    """Independent linear solve, used only to verify DP."""
    n = env.observation_space.n
    transition_matrix = np.zeros((n, n))
    rewards = np.zeros(n)
    for s in range(n):
        if s not in env.terminals:
            for p, sn, r, term in env.P[s][policy[s]]:
                rewards[s] += p * r
                if not term:
                    transition_matrix[s, sn] += p
    return np.linalg.solve(np.eye(n) - gamma * transition_matrix, rewards)

# SECTION: experience learning
class ExperienceOnly:
    """Restrict the learner's interface to sampled experience."""
    __slots__ = ("__env",)
    def __init__(self, env):
        self.__env = env
    def reset(self, seed=None):
        return self.__env.reset(seed=seed)
    def step(self, action):
        return self.__env.step(action)


def monte_carlo(env, policy, n_states=25, episodes=20_000,
                alpha=0.02, gamma=0.95, seed=42, checkpoints=(1000, 5000, 20000)):
    V = np.zeros(n_states)
    visits = np.zeros(n_states, dtype=int)
    history = np.zeros(episodes)
    snapshots = {}
    for ep in range(1, episodes + 1):
        s, _ = env.reset(seed=seed if ep == 1 else None)
        trajectory = []
        for _ in range(10_000):
            sn, r, terminated, truncated, _ = env.step(int(policy[s]))
            if truncated:
                raise RuntimeError("MC needs a complete episode")
            trajectory.append((s, r))
            s = sn
            if terminated:
                break
        else:
            raise RuntimeError("MC episode exceeded diagnostic guard")
        returns = np.zeros(len(trajectory))
        G = 0.0
        for t in reversed(range(len(trajectory))):
            G = trajectory[t][1] + gamma * G
            returns[t] = G
        for (s, _), G in zip(trajectory, returns):
            visits[s] += 1
            V[s] += alpha * (G - V[s])
        history[ep - 1] = V[0]
        if ep in checkpoints:
            snapshots[ep] = V.copy()
    return V, snapshots, history, visits


def td_zero(env, policy, n_states=25, episodes=20_000,
            alpha=0.02, gamma=0.95, seed=42, checkpoints=(1000, 5000, 20000)):
    V = np.zeros(n_states)
    visits = np.zeros(n_states, dtype=int)
    history = np.zeros(episodes)
    snapshots = {}
    for ep in range(1, episodes + 1):
        s, _ = env.reset(seed=seed if ep == 1 else None)
        for _ in range(10_000):
            sn, r, terminated, truncated, _ = env.step(int(policy[s]))
            if truncated:
                raise RuntimeError("Unexpected truncation in this experiment")
            target = r if terminated else r + gamma * V[sn]
            visits[s] += 1
            V[s] += alpha * (target - V[s])
            s = sn
            if terminated:
                break
        else:
            raise RuntimeError("TD episode exceeded diagnostic guard")
        history[ep - 1] = V[0]
        if ep in checkpoints:
            snapshots[ep] = V.copy()
    return V, snapshots, history, visits

# SECTION: experiment and reporting
def run_experiment(episodes=20_000, seed=42, alpha=0.02):
    rover = MarsRoverEnv()
    V_dp, policy = value_iteration(rover)
    residual = max(abs(V_dp[s] - max(action_values(rover, s, V_dp, GAMMA)))
                   for s in range(25) if s not in rover.terminals)
    assert residual < 1e-6
    assert np.max(np.abs(V_dp - exact_policy_value(rover, policy))) < 2e-5
    V_dry, policy_dry = value_iteration(MarsRoverEnv(slip=0))
    assert np.isclose(V_dry[0], -sum(GAMMA**t for t in range(7)) + 10*GAMMA**7)
    mc = monte_carlo(ExperienceOnly(MarsRoverEnv()), policy, episodes=episodes, seed=seed, alpha=alpha)
    td = td_zero(ExperienceOnly(MarsRoverEnv()), policy, episodes=episodes, seed=seed, alpha=alpha)
    assert np.array_equal(mc[3], td[3])
    V_drone, drone_policy = value_iteration(WindyDroneEnv())
    metrics = {"dp_start": float(V_dp[0]), "dry_start": float(V_dry[0]),
               "bellman_residual": float(residual), "drone_start": float(V_drone[0]),
               "config": {"episodes": episodes, "seed": seed, "alpha": alpha, "gamma": GAMMA,
                          "slip": 0.1, "paired_trajectories": True, "start_state": 0},
               "versions": {"python": platform.python_version(), "gymnasium": gym.__version__,
                            "numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    for name, result in [("MC", mc), ("TD", td)]:
        tail = result[2][-min(5000, episodes):]
        hits = []
        if episodes >= 500:
            rmse = np.sqrt(np.convolve((result[2]-V_dp[0])**2, np.ones(500)/500, mode="valid"))
            hits = np.flatnonzero(rmse < 0.5)
        metrics[name] = {"final": float(result[0][0]), "tail_std": float(tail.std()),
                         "tail_rmse": float(np.sqrt(np.mean((tail-V_dp[0])**2))),
                         "first_500_episode_rmse_below_0.5": int(hits[0]+500) if len(hits) else None}
    return dict(dp=V_dp, policy=policy, no_slip=V_dry, policy_dry=policy_dry,
                mc=mc, td=td, drone=V_drone, drone_policy=drone_policy, metrics=metrics)


def plot_convergence(data, output):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    count = len(data["mc"][2])
    for name, key, color in [("MC", "mc", "#b45309"), ("TD(0)", "td", "#0369a1")]:
        for ax in axes:
            ax.plot(np.arange(1, count+1), data[key][2], label=name, color=color, alpha=0.8, linewidth=0.7)
    for ax in axes:
        ax.axhline(data["dp"][0], linestyle="--", color="#111827", label="DP reference")
        ax.set(xlabel="Completed episodes", ylabel="Estimated V(start)")
        ax.grid(alpha=0.18)
    axes[0].set_title(f"Fixed policy, alpha = {data['metrics']['config']['alpha']}")
    axes[1].set(title="Early learning", xlim=(1, min(1500, count)))
    axes[0].legend()
    fig.savefig(Path(output) / "convergence.png", dpi=160)
    return fig


def save_results(data, output="results"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(data["metrics"], indent=2)+"\n", encoding="utf-8")
    checkpoints = {key: {str(ep): values.reshape(5, 5).tolist() for ep, values in data[key][1].items()}
                   for key in ("mc", "td")}
    (output / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2)+"\n", encoding="utf-8")
    np.savez(output / "values.npz", dp=data["dp"], policy=data["policy"], no_slip=data["no_slip"],
             mc=data["mc"][0], td=data["td"][0], mc_history=data["mc"][2], td_history=data["td"][2],
             visits=data["mc"][3], drone=data["drone"], drone_policy=data["drone_policy"])
    np.savetxt(output / "visits.csv", data["mc"][3].reshape(5, 5), delimiter=",", fmt="%d")
    return plot_convergence(data, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    if args.episodes < 1 or not 0 < args.alpha <= 1:
        parser.error("episodes must be positive and alpha must lie in (0, 1]")
    data = run_experiment(args.episodes, args.seed, args.alpha)
    figure = save_results(data, args.output)
    plt.close(figure)
    print(json.dumps(data["metrics"], indent=2))
