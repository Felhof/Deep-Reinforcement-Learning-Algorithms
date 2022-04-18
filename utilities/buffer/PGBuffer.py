from typing import Tuple

import numpy as np

from utilities.buffer.AbstractBuffer import AbstractBuffer
from utilities.config import Config


class PGBuffer(AbstractBuffer):
    def __init__(self: "PGBuffer", config: Config, buffer_size: int) -> None:
        super().__init__(config, buffer_size)

    def add_step_data(
        self: "PGBuffer",
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        pass

    def get_data(
        self: "PGBuffer",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        pass
