import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    scores = np.asarray(scores)
    row, col = scores.shape[-2], scores.shape[-1]

    mask_vals = np.triu(np.ones((row, col), dtype=bool), k=1)

    return np.where(mask_vals, mask_value, scores)
    pass