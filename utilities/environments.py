from collections import deque
from typing import Any, cast, SupportsFloat, Tuple

import cv2
import gymnasium as gym
import numpy as np
from utilities.types import ObservationDim


def _get_observation_dim_from_environment(environment: gym.Env) -> ObservationDim:
    shape = environment.observation_space.shape
    assert (
            len(shape) == 1 or len(shape) == 3
    ), "Only 1 or 3 dimensional observations supported"
    if len(shape) == 1:
        return shape[0]
    return cast(Tuple[int, int, int], shape)


class BaseEnvironmentWrapper:
    def __init__(self: "BaseEnvironmentWrapper", environment: gym.Env) -> None:
        self.environment = environment
        self.action_space = self.environment.action_space
        self.action_dim = 1
        self.reset = environment.reset
        self.observation_dim: ObservationDim = _get_observation_dim_from_environment(
            environment
        )
        if isinstance(
                self.environment.action_space, gym.spaces.Box
        ) and self.environment.action_space.shape == (1,):
            self.step = self.continuous_step
            self.number_of_actions = 1
            self.action_type = "Continuous"
        elif isinstance(self.environment.action_space, gym.spaces.Box):
            self.step = environment.step
            self.number_of_actions = self.environment.action_space.shape[0]
            self.action_type = "Continuous"
        else:
            self.step = environment.step
            self.number_of_actions = self.environment.action_space.n
            self.action_type = "Discrete"

    def continuous_step(
            self: "BaseEnvironmentWrapper", action: np.ndarray
    ) -> Tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        gym_action = np.array([action])
        return self.environment.step(gym_action)


class AtariWrapper(BaseEnvironmentWrapper):
    def __init__(
            self: "AtariWrapper",
            environment: gym.Env,
            width=84,
            height=84,
            greyscale=True,
            max_frames=4,
    ) -> None:
        super().__init__(environment)
        self.environment = environment
        self.observation_dim = (max_frames, height, width)
        self.width = width
        self.height = height
        self.greyscale = greyscale
        self.frames: deque = self._initialize_frame_stack(max_frames=max_frames)
        self.reset = self._reset
        self.step = self._step

    def _initialize_frame_stack(self: "AtariWrapper", max_frames: int = 4) -> deque:
        color_channels = 1 if self.greyscale else 3
        return deque(
            [np.zeros((color_channels, self.height, self.width)) for _ in range(max_frames)],
            maxlen=max_frames
        )

    def _preprocess_observation(self: "AtariWrapper", obs: Any) -> Any:
        if self.greyscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.greyscale:
            obs = np.expand_dims(obs, 0)
        self.frames.append(obs)
        return np.concatenate(self.frames)

    def _reset(self: "AtariWrapper") -> Tuple[Any, dict[str, Any]]:
        frame, info = self.environment.reset()
        obs = self._preprocess_observation(frame)
        return obs, info

    def _step(
            self: "AtariWrapper", action: np.ndarray[Any, Any]
    ) -> Tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        frame, reward, terminated, truncated, info = self.environment.step(action)
        obs = self._preprocess_observation(frame)
        return obs, reward, terminated, truncated, info
