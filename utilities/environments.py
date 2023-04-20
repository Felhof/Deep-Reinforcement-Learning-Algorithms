from collections import deque
from typing import Any, cast, SupportsFloat, Tuple

import cv2
import gymnasium as gym
import numpy as np
from utilities.LazyFrames import LazyFrames
from utilities.types.environment import Info, Observation
from utilities.types.types import ObservationDim


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
        self.is_really_done = True
        if isinstance(
            self.environment.action_space, gym.spaces.Box
        ) and self.environment.action_space.shape == (1,):
            self.step = self.continuous_step
            self.number_of_actions = 1
            self.action_type = "Continuous"
        elif isinstance(self.environment.action_space, gym.spaces.Box):
            self.step = self._step
            self.number_of_actions = self.environment.action_space.shape[0]
            self.action_type = "Continuous"
        else:
            self.step = self._step
            self.number_of_actions = self.environment.action_space.n
            self.action_type = "Discrete"

    def _step(
        self: "BaseEnvironmentWrapper", action: np.ndarray
    ) -> Tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.environment.step(action)
        self.is_really_done = terminated or truncated
        return obs, reward, terminated, truncated, info

    def continuous_step(
        self: "BaseEnvironmentWrapper", action: np.ndarray
    ) -> Tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        gym_action = np.array([action])
        return self._step(gym_action)

    def true_reset(self: "BaseEnvironmentWrapper") -> Tuple[Observation, Info]:
        return self.reset()


class AtariWrapper(BaseEnvironmentWrapper):
    """
    Combines elements of the wrappers from Stable Baselines 3: https://stable-baselines3.readthedocs.io/en/v1.0/_modules/stable_baselines3/common/atari_wrappers.html
    and OpenAI: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/atari_wrappers.py
    In particular
        - Warps frames to greyscale and 84x84.
        - Ends episode but does not reset environment on loss of live.
        - Fires on reset for environments that are fixed until firing.
        - Clips reward to between -1 and 1.
        - Uses memory-efficient lazy frames for observations.
    """  # noqa

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
        self.is_fire_reset_env = (
            environment.unwrapped.get_action_meanings()[1] == "FIRE"
        )
        self.lives = 0

    def _initialize_frame_stack(self: "AtariWrapper", max_frames: int = 4) -> deque:
        color_channels = 1 if self.greyscale else 3
        return deque(
            [
                np.zeros((color_channels, self.height, self.width))
                for _ in range(max_frames)
            ],
            maxlen=max_frames,
        )

    def _preprocess_observation(self: "AtariWrapper", obs: np.ndarray) -> LazyFrames:
        if self.greyscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.greyscale:
            obs = np.expand_dims(obs, 0)
        self.frames.append(obs)
        return LazyFrames(self.frames)

    def _reset(self: "AtariWrapper") -> Tuple[LazyFrames, dict[str, Any]]:
        if self.is_really_done:
            obs, info = self.true_reset(with_noops=True)
        else:
            frame, _, _, _, info = self.environment.step(0)
            if self.is_fire_reset_env:
                frame, _, _, _, info = self.environment.step(1)
            obs = self._preprocess_observation(frame)

        self.lives = self.environment.unwrapped.ale.lives()
        return obs, info

    def _step(
        self: "AtariWrapper", action: np.ndarray[Any, Any]
    ) -> Tuple[LazyFrames, SupportsFloat, bool, bool, dict[str, Any]]:
        frame, reward, terminated, truncated, info = self.environment.step(action)
        info["True Reward"] = reward
        obs = self._preprocess_observation(frame)
        reward = np.clip(reward, -1.0, 1.0)

        self.is_really_done = terminated or truncated
        lives = self.environment.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives

        return obs, reward, terminated, truncated, info

    def true_reset(self: "AtariWrapper", with_noops=False) -> Tuple[LazyFrames, Info]:
        number_of_noops = np.random.randint(0, 31) if with_noops else 0

        def reset():
            frame, info = self.environment.reset()
            if self.is_fire_reset_env:
                while True:
                    frame, info, terminated, truncated, _ = self.environment.step(1)
                    if not (terminated or truncated):
                        break
                    self.environment.reset()

            return frame, info

        frame, info = reset()

        for _ in range(number_of_noops):
            frame, info, terminated, truncated, _ = self.environment.step(0)

            if terminated or truncated:
                frame, info = reset()

        obs = self._preprocess_observation(frame)

        return obs, info

    def render(self):
        self.environment.render()
