"""Check the published numerical claims after a baseline notebook run."""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parent
m = json.loads((root / "results/metrics.json").read_text())
a = np.load(root / "results/values.npz")
assert m["config"]["episodes"] == 20_000 and m["config"]["seed"] == 42
assert abs(m["dp_start"] - (-1.4161242807269079)) < 1e-6
assert abs(m["dry_start"] - 0.9501188828124982) < 1e-6
assert abs(m["drone_start"] - 4.301344159038449) < 1e-6
assert abs(m["MC"]["final"] - (-1.8889009268977217)) < 1e-6
assert abs(m["TD"]["final"] - (-1.396947526273843)) < 1e-6
assert abs(m["MC"]["tail_std"] - 0.3368539696274805) < 1e-6
assert abs(m["TD"]["tail_std"] - 0.10413190611654952) < 1e-6
assert m["MC"]["first_500_episode_rmse_below_0.5"] == 500
assert m["TD"]["first_500_episode_rmse_below_0.5"] == 1203
assert a["visits"][13] == 16
assert np.isfinite(a["mc_history"]).all() and len(a["mc_history"]) == 20_000
print("Published baseline claims verified")
