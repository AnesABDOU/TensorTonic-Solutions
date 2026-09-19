import numpy as np

def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    """
    x = np.asarray(x)
    h_prev = np.asarray(h_prev)
    single_sample = x.ndim == 1
    if single_sample:
        x = x.reshape(1, -1)
        h_prev = h_prev.reshape(1, -1)

    Wz, Uz, bz = params["Wz"], params["Uz"], params["bz"]
    Wr, Ur, br = params["Wr"], params["Ur"], params["br"]
    Wh, Uh, bh = params["Wh"], params["Uh"], params["bh"]
    
    def sigmoid(p):
        return 1 / (1 + np.exp(-p))

    zt = sigmoid(x @ Wz + h_prev @ Uz + bz)

    rt = sigmoid(x @ Wr + h_prev @ Ur + br)

    ht_hat = np.tanh(x @ Wh + (rt * h_prev) @ Uh + bh)

    ht = (1.0 - zt) * h_prev + zt * ht_hat

    return ht[0] if single_sample else ht
    
    pass