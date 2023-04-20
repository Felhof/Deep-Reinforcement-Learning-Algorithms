from typing import Any, Dict, Union

import numpy as np
from utilities.LazyFrames import LazyFrames

Info = Dict[str, Any]
Observation = Union[np.ndarray, LazyFrames]
