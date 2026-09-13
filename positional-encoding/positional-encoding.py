import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div   = 1.0 / (base**(np.arange(0, d_model, 2) / d_model))

    res = position*div
    pe[:, 0::2] = np.sin(res)
    pe[:, 1::2] = np.cos(res[:, :d_model//2])
    return pe
    pass