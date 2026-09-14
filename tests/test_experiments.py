import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from rl_experiments import (MarsRoverEnv, WindyDroneEnv, ExperienceOnly, value_iteration,
                            exact_policy_value, action_values, monte_carlo, td_zero)

@pytest.mark.parametrize("factory", [MarsRoverEnv, WindyDroneEnv])
def test_environment_contract(factory):
    env = factory()
    check_env(env, skip_render_check=True)
    for s, actions in env.P.items():
        for tr in actions.values():
            assert np.isclose(sum(t[0] for t in tr), 1)
            assert all(p > 0 and env.observation_space.contains(sn) for p, sn, _, _ in tr)
            if s in env.terminals:
                assert tr == [(1.0, s, 0.0, True)]

def test_wall_merging_and_terminal_arrival():
    env = MarsRoverEnv()
    assert env.P[0][0] == [(0.9, 0, -1.0, False), (0.1, 1, -1.0, False)]
    assert (0.8, 24, 10.0, True) in env.P[23][1]
    assert (0.8, 12, -10.0, True) in env.P[11][1]

@pytest.mark.parametrize("factory", [MarsRoverEnv, WindyDroneEnv])
def test_dp_against_independent_linear_solve(factory):
    env = factory()
    values, policy = value_iteration(env)
    assert np.max(np.abs(values - exact_policy_value(env, policy))) < 2e-5
    assert max(abs(values[s] - max(action_values(env, s, values, 0.95)))
               for s in range(len(values)) if s not in env.terminals) < 1e-6

def test_deterministic_shortest_safe_path():
    values, _ = value_iteration(MarsRoverEnv(slip=0))
    assert np.isclose(values[0], -sum(0.95**t for t in range(7)) + 10*0.95**7)

class TwoStep:
    def reset(self, seed=None):
        self.t = 0
        return 0, {}
    def step(self, action):
        self.t += 1
        return (1, -1.0, False, False, {}) if self.t == 1 else (2, 10.0, True, False, {})

def test_mc_returns_and_td_update_timing():
    policy = np.array([0, 0, -1])
    assert np.allclose(monte_carlo(TwoStep(), policy, n_states=3, episodes=1, alpha=1)[0], [8.5, 10, 0])
    assert np.allclose(td_zero(TwoStep(), policy, n_states=3, episodes=1, alpha=1)[0], [-1, 10, 0])

class TerminalToValuedState:
    """Revisit state 0 as a terminal observation, to expose an erroneous bootstrap."""
    def reset(self, seed=None):
        return 0, {}
    def step(self, action):
        return 0, -10.0, True, False, {}

def test_terminal_target_does_not_bootstrap():
    values = td_zero(TerminalToValuedState(), np.array([0]), n_states=1, episodes=2, alpha=1)[0]
    assert values[0] == -10

def test_paired_streams_and_no_model_access():
    _, policy = value_iteration(MarsRoverEnv())
    adapter = ExperienceOnly(MarsRoverEnv())
    assert not hasattr(adapter, "P")
    mc = monte_carlo(adapter, policy, episodes=50, checkpoints=(10, 50))
    td = td_zero(ExperienceOnly(MarsRoverEnv()), policy, episodes=50, checkpoints=(10, 50))
    assert np.array_equal(mc[3], td[3])
    assert set(mc[1]) == set(td[1]) == {10, 50}

class Truncated:
    def reset(self, seed=None):
        return 0, {}
    def step(self, action):
        return 0, -1.0, False, True, {}

@pytest.mark.parametrize("learner", [monte_carlo, td_zero])
def test_truncation_is_not_silently_treated_as_termination(learner):
    with pytest.raises(RuntimeError):
        learner(Truncated(), np.array([0]), n_states=1, episodes=1)
