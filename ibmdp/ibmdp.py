from gymnasium import Env, Wrapper, spaces
import numpy as np

def check_bounded_obs(observation_space):
    idx_unbounded_low, idx_unbounded_high = [], []
    for idx, bound in enumerate(observation_space.low):
        if bound <= -1e20:
            idx_unbounded_low.append(idx)
        if observation_space.high[idx] >= 1e20:
            idx_unbounded_high.append(idx)
    return (idx_unbounded_low, idx_unbounded_high)

def bound_observation_space(observation_space, bounding_value):
    idx_unbounded_low, idx_unbounded_high = check_bounded_obs(observation_space)
    low = observation_space.low.copy()
    high = observation_space.high.copy()

    for idx in idx_unbounded_low:
        low[idx] = -bounding_value
    for idx in idx_unbounded_high:
        high[idx] = bounding_value
    return low, high


class IBMDP(Env):
    """
    Generic class to make Iterative Bounding MDPs [1]_.

    Parameters
    ----------
    env: gymnasium.Env
        A gymnasium environment. This environment should represent a factored MDP.
        Continuous observations, discrete actions.
    
    zeta: float
        The penalty for taking an information gathering action. See [1]_.

    info_gathering_actions: list of tuples.
        each tuple represent an info gathering actions (feature, value): feat<= val ?

    bounding_value: float
        When the observation space is unbounded, it is clipped between -boudning_value, bounding_value

    References
    ----------
    .. [1] N. Topin et. al. : Iterative Bounding MDPs: Learning Interpretable Policies via Non-Interpretable Methods https://arxiv.org/abs/2102.13045
 
    """
    def __init__(self, env: Env, zeta: float, info_gathering_actions: list, bounding_value: float=10):
        # Do something with reward range?
        assert isinstance(env.observation_space, spaces.Box) and isinstance(env.action_space, spaces.Discrete), "Env is not a factored MDP!"
        assert all([aig[0] < env.observation_space.shape[0] for aig in info_gathering_actions]), "Some AIG are testing non-existing features."
        assert all([isinstance(aig[1], float) or isinstance(aig[1], int)  for aig in info_gathering_actions]), "Some AIG are testing non-float values"
        assert isinstance(zeta, float) or isinstance(zeta, int), "Zeta should be a float value"
            
        self.nb_base_actions = env.action_space.n
        self.nb_base_features = env.observation_space.shape[0]

        low, high = bound_observation_space(env.observation_space, bounding_value)
        self.init_bounds_ = np.append(low, high)
        self.base_env = env

        self.observation_space = spaces.Box(low = np.tile(low, 3), high = np.tile(high, 3))
        
        self.action_space = spaces.Discrete(env.action_space.n, len(info_gathering_actions))
        self.info_actions = info_gathering_actions

        if hasattr(env, "reward_range"):
            self.zeta = np.clip(zeta, *env.reward_range)
        else:
            self.zeta = zeta
        
    def reset(self, seed=None):
        s, infos = self.base_env.reset()
        self._state = np.append(s, self.init_bounds_)

        self._infos = infos
        self._infos["partial_obs"] = self.init_bounds_
        self._infos["depth"] = 0

        return self._state, self._infos
    
    def step(self, action):
        if action < self.nb_base_actions:
            s, r, term, trunc, infos = self.base_env.step(action)
            self._infos.update(infos)
            self._state = np.append(s, self.init_bounds_)
        else:
            r = self.zeta
            term = trunc = False
            feature, value = self.info_actions[action-self.nb_base_actions]
            if self._state[feature] <= value:
                self._state[2 * self.nb_base_features + feature] = min(value, self._state[2 * self.nb_base_features + feature])
            else:
                self._state[self.nb_base_features + feature] = max(value, self._state[self.nb_base_features + feature])
            self._infos["partial_observation"] = self._state[-2 * self.nb_base_features : ]
            self._infos["depth"] += 1
        
        return self._state, r, term, trunc, self._infos