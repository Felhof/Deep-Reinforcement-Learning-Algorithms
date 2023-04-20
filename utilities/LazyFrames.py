from collections import deque

import numpy as np


class LazyFrames:
    """
    Basically a minimalist implementation of LazyFrames from OpenAI's Atari Wrapper:
    https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/atari_wrappers.py#L229
    """

    def __init__(self, frames: deque) -> None:
        self.frames = list(frames)

    def __array__(self, dtype=None):
        array = np.concatenate(self.frames, axis=0)
        if dtype is not None:
            array = array.astype(dtype)
        return array
