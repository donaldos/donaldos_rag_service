from __future__ import annotations
from typing import List
import numpy as np

def l2_normalize(vec: List[float]) -> List[float]:
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 0 else v.tolist()
