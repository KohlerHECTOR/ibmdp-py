from stable_baselines3 import PPO
from gymnasium import make
from gymnasium.wrappers.time_limit import TimeLimit
from ibmdp import IBMDP

env = make("CartPole-v1")
env = IBMDP(env, zeta=0, info_gathering_actions=[(0,0)])
env = TimeLimit(env, 1000)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(1e5)