from typing import Any, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np


class EnvironmentWrapper:
    def __init__(self: "EnvironmentWrapper", environment: gym.Env) -> None:
        self.environment = environment
        self.reset = environment.reset
        if isinstance(
                self.environment.action_space, gym.spaces.Box
        ) and self.environment.action_space.shape == (1,):
            self.step = self.continuous_step
        else:
            self.step = environment.step

    def continuous_step(
            self: "EnvironmentWrapper", action: np.ndarray
    ) -> Tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        gym_action = np.array([action])
        return self.environment.step(gym_action)
