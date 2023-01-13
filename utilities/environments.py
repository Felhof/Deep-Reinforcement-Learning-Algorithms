from typing import Any, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np


class EnvironmentWrapper:
    def __init__(self: "EnvironmentWrapper", environment: gym.Env) -> None:
        self.environment = environment
        self.action_space = self.environment.action_space
        self.reset = environment.reset
        self.observation_dim = environment.observation_space.shape
        if isinstance(
                self.environment.action_space, gym.spaces.Box
        ) and self.environment.action_space.shape == (1,):
            self.step = self.continuous_step
            self.number_of_actions = 1
            self.action_type = "Continuous"
        elif isinstance(
                self.environment.action_space, gym.spaces.Box
        ):
            self.step = environment.step
            self.number_of_actions = self.environment.action_space.shape[0]
            self.action_type = "Continuous"
        else:
            self.step = environment.step
            self.number_of_actions = self.environment.action_space.n
            self.action_type = "Discrete"

    def continuous_step(
            self: "EnvironmentWrapper", action: np.ndarray
    ) -> Tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        gym_action = np.array([action])
        return self.environment.step(gym_action)
