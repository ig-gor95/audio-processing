from dataclasses import dataclass
from typing import List, Dict

import numpy as np


@dataclass
class DiarizedResult:
    valid_segments: List[Dict[str, float]]
    labels: np.ndarray