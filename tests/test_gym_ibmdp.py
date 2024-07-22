import gymnasium as gym
from ibmdp import IBMDP
import pytest

def test():
    env = gym.make("CartPole-v1")
    aigs = [(0, 0), (1, 0)]
    env = IBMDP(env, zeta=1, info_gathering_actions=aigs)
    s, infos = env.reset()
    for _ in range(1000):
        s, r, term, trunc, infos = env.step(env.action_space.sample())
        if term or trunc:
            s, infos = env.reset()

@pytest.mark.xfail(raises=AssertionError)
@pytest.mark.parametrize("env", ["Pendulum-v1", "CartPole-v1"])
@pytest.mark.parametrize("aigs", [[(0, 0)], [(10_000, 0)], [(0, "ii")]])
@pytest.mark.parametrize("zeta", [0, 10_000, "i"])
def test_assertion_errors(env, aigs, zeta):
    env = gym.make(env) # continuous actions
    env = IBMDP(env, zeta, info_gathering_actions=aigs)
    