import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    counts = np.unique(y, return_counts=True)
    p = counts[1] / len(y)
    entropy = -sum(p * np.log(p)/np.log(2))
    return float(entropy)
    pass