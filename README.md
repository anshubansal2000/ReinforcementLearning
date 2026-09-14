# Reinforcement Learning, Explained Through Experiments

**A Mars rover. A windy drone. Three ways to estimate the value of a decision.**

We built two small Gymnasium environments and compared dynamic programming, Monte Carlo,
and TD(0). The experiments expose how learning targets, uncertainty, and data coverage affect
what an agent learns. No GPU is required.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anshubansal2000/ReinforcementLearning/blob/main/mars_rover_assignment.ipynb)
[![Verify experiments](https://github.com/anshubansal2000/ReinforcementLearning/actions/workflows/verify.yml/badge.svg)](https://github.com/anshubansal2000/ReinforcementLearning/actions/workflows/verify.yml)

**[Read the article →](blog/reinforcement-learning.md)** ·
**[Browse the executed notebook →](mars_rover_assignment.ipynb)** ·
**[Inspect the Python implementation →](rl_experiments.py)**

![MC and TD start-state estimates against the DP reference](results/convergence.png)

*One paired run: 20,000 episodes per method, seed 42, alpha 0.02. Left: full training;
right: early learning. The dashed line is the DP reference.*

## What we found

| Measurement | Result |
|---|---:|
| Optimal rover start value, with slip | −1.416 |
| Optimal rover start value, without slip | +0.950 |
| MC final start estimate | −1.889 |
| TD final start estimate | −1.397 |
| MC / TD late start-value standard deviation | 0.337 / 0.104 |
| Visits to state 13 after 20,000 episodes | 16 |
| Optimal windy-drone start value | 4.301 |

TD was steadier near the start late in training. MC reached the chosen early error threshold
sooner. Both struggled in rarely visited states. This is one paired-seed comparison, not a
universal algorithm ranking. DP performs control; MC and TD evaluate its fixed stochastic-world
policy. The standard deviations describe consecutive estimates, not independent-seed uncertainty.

## Start in Google Colab

1. Click **Open in Colab** above.
2. Save a copy in Google Drive if you want to edit and retain your work.
3. Use a CPU runtime and run cells from top to bottom, starting with the installation cell.
4. Inspect the value grids, visit counts, and plots. Use the final optional cell to download results.

The notebook embeds all implementation code and requires no local files or repository clone.
If Colab restarts, run from the top again: old outputs may remain while Python's memory is empty.

## Run locally with uv

Python 3.12 is the reference runtime. Clone this repository, then:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run --no-project python -m pytest -q
uv run --no-project python run_notebook.py
uv run --no-project python verify_results.py
```

`run_notebook.py` uses the same interpreter for its fresh notebook kernel. It saves an executed
notebook, an HTML export, and result artifacts. Open `mars_rover_assignment.html` locally for a
readable copy without a notebook editor. GitHub's HTML file view shows source; use the notebook
or article for an online reading experience.

Prefer regular Python? Create and activate a virtual environment, run
`python -m pip install -r requirements.txt`, then the same three Python commands without `uv run`.
The requirements file pins direct dependencies; `results/environment.txt` records all package
versions from the reference run.

## Run or vary the experiment from the command line

```bash
uv run --no-project python rl_experiments.py
uv run --no-project python rl_experiments.py --seed 7 --alpha 0.01 --output results-seed7
```

The default reruns the published baseline and writes `results/`. Use another output directory
for variants. `verify_results.py` checks the published baseline only, not arbitrary variants.

## Follow the learning path

| Experiment | Question |
|---|---|
| Rover environment | What happens when an action can slip? |
| Value iteration | Can we calculate optimal decisions from a known model? |
| MC and TD | Can sampled experience recover a fixed policy's values? |
| Windy drone | Can the same DP algorithm handle different transition rules? |

Try repeated seeds, learning-rate schedules, or broader initial-state coverage after reproducing
the baseline. Report those changes explicitly. The article includes proposed extensions and
plain-language connections to Bellman, AlphaGo, TRPO, and PPO; those advanced systems were not
implemented in these experiments.

## Repository map

- [`blog/reinforcement-learning.md`](blog/reinforcement-learning.md): publication article.
- [`mars_rover_assignment.ipynb`](mars_rover_assignment.ipynb): self-contained, executed notebook.
- [`rl_experiments.py`](rl_experiments.py): environments, DP, MC, TD, reporting, and CLI.
- [`tests/test_experiments.py`](tests/test_experiments.py): API, probability, Bellman, and learning checks.
- [`results/metrics.json`](results/metrics.json): exact diagnostics and experiment configuration.
- [`results/checkpoints.json`](results/checkpoints.json): six checkpoint grids.
- [`results/visits.csv`](results/visits.csv): visits arranged as the rover's 5×5 grid.
- [`results/values.npz`](results/values.npz): values, policies, and learning histories.
- [`build_notebook.py`](build_notebook.py): regenerate the notebook after changing Python source.
- [`run_notebook.py`](run_notebook.py): execute and export using a fresh kernel.
- [`verify_results.py`](verify_results.py): confirm the article's baseline numerical claims.

For implementation changes, edit `rl_experiments.py`, run `build_notebook.py`, then rerun tests
and the notebook. Building removes saved notebook outputs; executing restores them. The GitHub
Actions workflow runs tests, the full notebook, and the published-result checks on pushes and PRs.

## Attribution and license

The rover task specification is adapted from [Vizuara AI Labs' RL in Production, Lecture 01
exercise](https://rl-bootcamp-decks.vercel.app/pdfs/lecture-01-assignment.pdf). This repository
contains our implementation, experiments, and interpretation of established RL methods. It does
not claim novelty of the algorithms or authorship of the original exercise. The source PDF and
course slides are linked, not redistributed.

See [NOTICE](NOTICE) for provenance and the existing [Apache-2.0 license](LICENSE) for repository
licensing. External course materials and referenced publications retain their respective rights.
