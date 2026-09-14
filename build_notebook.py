"""Build a standalone teaching notebook from the tested Python source."""
from pathlib import Path
import textwrap
import nbformat

ROOT = Path(__file__).resolve().parent
source = (ROOT / "rl_experiments.py").read_text(encoding="utf-8")
cells = []
def md(text):
    cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))
def code(text):
    cells.append(nbformat.v4.new_code_cell(textwrap.dedent(text).strip()))

md('''
# Reinforcement learning: a rover, a drone, and three ways to estimate value

We investigate what changes when an algorithm knows the world's dynamics versus learning
through experience. Run top to bottom in a fresh Python session. All implementation code is
embedded here; the notebook requires no repository clone or local data files.

[Read the article](https://github.com/anshubansal2000/ReinforcementLearning/blob/main/blog/reinforcement-learning.md)
· [Repository](https://github.com/anshubansal2000/ReinforcementLearning)

**Colab:** run the installation cell first. No GPU is required. Save a copy in Drive to preserve
your notebook. Runtime files are temporary; the final optional cell downloads the results.
If Colab resets, rerun from the top: visible old outputs do not mean definitions remain in memory.

**Local:** use the README's uv instructions and select the project's Python kernel.

Rover task specification adapted from [Vizuara AI Labs, Lecture 01](https://rl-bootcamp-decks.vercel.app/pdfs/lecture-01-assignment.pdf).
''')
code('''
import sys
if "google.colab" in sys.modules:
    %pip install -q "gymnasium==1.3.0" "numpy==2.1.3" "matplotlib==3.10.0"
''')
code(source.split("# SECTION: environments")[0])
code('''
np.set_printoptions(precision=2, suppress=True, linewidth=110)
print("Gymnasium:", gym.__version__)
frozen = gym.make("FrozenLake-v1", is_slippery=True)
print("reset:", frozen.reset(seed=0))
print("step 5-tuple:", frozen.step(1))
frozen.close()
''')
md('''
## Experiment 1 · Define the world

States are the 25 grid cells, encoded as `row*5+column`. Actions are 0 up, 1 right,
2 down, 3 left. Intended direction: 80%; each perpendicular slip: 10%. Walls clip movement.
Entering the goal (4,4) gives +10; entering either crater (1,3) or (2,2) gives -10.
Other moves cost -1. Goal and craters terminate the episode. Arrival rewards replace the
ordinary move cost; terminal states have zero remaining reward.

`reset` starts an episode; `step` samples one outcome. `P` exposes the complete model for DP.
''')
env_source = source.split("# SECTION: environments\n")[1].split("# SECTION: dynamic programming")[0]
rover_source, drone_source = env_source.split("class WindyDroneEnv", 1)
code(rover_source)
code('''
from gymnasium.utils.env_checker import check_env
rover = MarsRoverEnv()
check_env(rover, skip_render_check=True)
assert all(np.isclose(sum(t[0] for t in tr), 1)
           for actions in rover.P.values() for tr in actions.values())
print("Gymnasium API and model probability checks passed")
print("Possible outcomes of requesting right at state 23:", rover.P[23][1])

if "MarsRover-v0" not in gym.registry:
    gym.register(id="MarsRover-v0", entry_point=MarsRoverEnv)
demo = gym.make("MarsRover-v0")
demo.action_space.seed(7)
s, _ = demo.reset(seed=7)
print("Random episode: (state, action, reward, next_state)")
for steps in range(10_000):
    a = int(demo.action_space.sample())
    result = demo.step(a)
    assert len(result) == 5
    sn, reward, terminated, truncated, _ = result
    print((s, a, reward, sn))
    s = sn
    if terminated or truncated:
        break
else:
    raise RuntimeError("Episode exceeded diagnostic guard")
assert terminated and s in demo.unwrapped.terminals
print("Ended after", steps+1, "steps at", rc(s))
demo.close()
''')
md('''
## Experiment 2 · Calculate values using the model

Bellman's idea: evaluate the immediate reward and the value of what comes next. For each
action, average across possible outcomes; then take the best action. At state 23 with zero
initial values, requesting right gives `0.8*10 + 0.1*(-1) + 0.1*(-1) = 7.8`.

Value iteration repeats in-place sweeps until the largest change is below 1e-6. It performs
control: the extracted greedy policy is optimal up to numerical tolerance. We check against
an independent linear-equation evaluation of that policy.
''')
code(source.split("# SECTION: dynamic programming\n")[1].split("# SECTION: experience learning")[0])
code('''
V_dp, policy = value_iteration(rover)
print("Optimal stochastic values:")
print(V_dp.reshape(5, 5))
print("Policy (X = crater, G = goal):")
for row in policy_grid(policy, (5, 5), rover.terminals, rover.goal):
    print(" ".join(row))
residual = max(abs(V_dp[s] - max(action_values(rover, s, V_dp, GAMMA)))
               for s in range(25) if s not in rover.terminals)
assert residual < 1e-6
assert np.max(np.abs(V_dp - exact_policy_value(rover, policy))) < 2e-5
print("Bellman residual:", residual)
print("DP start value:", V_dp[0])

dry_rover = MarsRoverEnv(slip=0)
V_dry, policy_dry = value_iteration(dry_rover)
print("No-slip values:")
print(V_dry.reshape(5, 5))
print("No-slip policy:")
for row in policy_grid(policy_dry, (5, 5), dry_rover.terminals, dry_rover.goal):
    print(" ".join(row))
print("No-slip minus stochastic values:")
print((V_dry - V_dp).reshape(5, 5))
assert np.isclose(V_dry[0], -sum(GAMMA**t for t in range(7)) + 10*GAMMA**7)
''')
md('''
Removing slip raises the start value from -1.416 to 0.950. Adjacent to a crater, reliable
movement avoids accidental entries. It also avoids detours and wasted moves. Terminal cells
remain zero because their arrival rewards were already paid. Keep the stochastic policy
for the next comparison.

## Experiment 3 · Learn by driving

MC and TD evaluate that fixed policy; neither changes the chosen actions. Both start at zero,
use alpha 0.02, gamma 0.95, and seed 42, and receive identical trajectories. Only reset and step
are exposed to the learners. MC computes returns backward, then updates every visit in time
order. TD updates after every step. Unexpected truncation or episodes longer than 10,000 steps
raise errors, avoiding silently biased incomplete returns.
''')
code(source.split("# SECTION: experience learning\n")[1].split("# SECTION: experiment and reporting")[0])
code('''
class TwoStepExperience:
    def reset(self, seed=None):
        self.t = 0
        return 0, {}
    def step(self, action):
        self.t += 1
        return (1, -1.0, False, False, {}) if self.t == 1 else (2, 10.0, True, False, {})

test_policy = np.array([0, 0, -1])
assert np.allclose(monte_carlo(TwoStepExperience(), test_policy, n_states=3,
                             episodes=1, alpha=1)[0], [8.5, 10, 0])
assert np.allclose(td_zero(TwoStepExperience(), test_policy, n_states=3,
                         episodes=1, alpha=1)[0], [-1, 10, 0])
print("Hand-calculated MC and TD checks passed")

mc = monte_carlo(ExperienceOnly(MarsRoverEnv()), policy)
td = td_zero(ExperienceOnly(MarsRoverEnv()), policy)
assert np.array_equal(mc[3], td[3])
for name, result in [("MC", mc), ("TD(0)", td)]:
    for ep, values in result[1].items():
        print(f"\\n{name}, {ep:,} episodes")
        print(values.reshape(5, 5))
    print("Visit counts:")
    print(result[3].reshape(5, 5))
''')
# Reporting functions are reusable; run_experiment is defined but not called by the notebook.
code(source.split("# SECTION: experiment and reporting\n")[1].split('if __name__ == "__main__":')[0])
code('''
data = {"dp": V_dp, "mc": mc, "td": td, "metrics": {"config": {"alpha": 0.02}}}
Path("results").mkdir(exist_ok=True)
plot_convergence(data, "results")
plt.show()

print("Start-state diagnostics (one paired seed):")
metrics = {"dp_start": float(V_dp[0]), "dry_start": float(V_dry[0]),
           "bellman_residual": float(residual),
           "config": {"episodes": 20000, "seed": 42, "alpha": 0.02, "gamma": GAMMA,
                      "slip": 0.1, "paired_trajectories": True, "start_state": 0},
           "versions": {"python": platform.python_version(), "gymnasium": gym.__version__,
                        "numpy": np.__version__, "matplotlib": matplotlib.__version__}}
for name, result in [("MC", mc), ("TD", td)]:
    tail = result[2][-5000:]
    rmse = np.sqrt(np.convolve((result[2]-V_dp[0])**2, np.ones(500)/500, mode="valid"))
    hits = np.flatnonzero(rmse < 0.5)
    metrics[name] = {"final": float(result[0][0]), "tail_std": float(tail.std()),
                     "tail_rmse": float(np.sqrt(np.mean((tail-V_dp[0])**2))),
                     "first_500_episode_rmse_below_0.5": int(hits[0]+500) if len(hits) else None}
    print(name, metrics[name])
''')
md('''
MC uses full returns, which can vary substantially with later slips and hazards. TD uses one
reward and a bootstrapped estimate; this often reduces sampling variability but creates dependence
on initially inaccurate estimates. Here, MC reaches a 500-episode rolling RMSE below 0.5 earlier,
while TD has lower start-value variability during the last 5,000 episodes. Constant alpha leaves
residual noise. A single paired run cannot establish a universal winner.

State 13 was visited only 16 times: its MC/TD estimates remain far from DP even after 20,000
episodes. An accurate start-state curve is not evidence of accurate values everywhere.

## Experiment 4 · A second world

Windy drone: S is a 3x4 grid; A is the four directions; P applies the requested action with
probability 0.8 or moves east with probability 0.2. R is +8 at delivery (2,3), -6 at a tower
(1,2), -0.5 otherwise. Gamma is 0.95. Both special cells terminate episodes; start is (0,0).
The same DP algorithm applies without modification.
''')
code("class WindyDroneEnv" + drone_source)
code('''
drone = WindyDroneEnv()
check_env(drone, skip_render_check=True)
V_drone, drone_policy = value_iteration(drone)
print("Drone optimal values:")
print(V_drone.reshape(3, 4))
for row in policy_grid(drone_policy, (3, 4), drone.terminals, drone.goal):
    print(" ".join(row))
assert np.max(np.abs(V_drone - exact_policy_value(drone, drone_policy))) < 2e-5
''')
md('''
The drone favors the top route from the start. Eastward wind aligns with travel across the top;
at the right wall a gust causes a delay, rather than a tower collision. Start value: about 4.301.

## What these experiments teach

1. A real rover may not know the complete transition model. MC and TD require sampled experience;
   model-based planning can still help when a useful approximate model is available.
2. MC updates after a completed episode; TD updates after each transition. On the transition into
   a crater the episode ends immediately, so both can update the preceding state at that boundary.
3. To improve a policy from experience, learn action values and explore, for example using
   epsilon-greedy Q-learning. DP already performed control here; MC and TD performed evaluation.

Next experiments: repeat across seeds, change alpha, and broaden the starting-state distribution.
Report these as changed experiments rather than silently altering the baseline.
''')
code('''
metrics["drone_start"] = float(V_drone[0])
data.update(policy=policy, no_slip=V_dry, policy_dry=policy_dry,
            drone=V_drone, drone_policy=drone_policy, metrics=metrics)
fig = save_results(data)
plt.close(fig)
print("Saved results/metrics.json, checkpoints.json, values.npz, visits.csv, convergence.png")
print("All notebook checks passed.")
''')
md('''
**Optional Colab download:** run this cell to download the result files before the runtime expires.
Use File → Download → Download .ipynb to save the executed notebook itself.
''')
code('''
if "google.colab" in sys.modules:
    import shutil
    from google.colab import files
    shutil.make_archive("rl-results", "zip", "results")
    files.download("rl-results.zip")
''')
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbformat.validate(nb)
nbformat.write(nb, ROOT / "mars_rover_assignment.ipynb")
print("Built standalone notebook")
