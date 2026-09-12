import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    
    N = len(seqs)
    
    if max_len is not None:
        L = max_len
    elif N > 0:
        L = max(len(s) for s in seqs)
    else:
        L = 0

    if N == 0:
        return np.empty((N, L), dtype=int)
    
    for i, s in enumerate(seqs):
        
        s_trimmed = s[:L]
        
        while len(s_trimmed) < L:
            s_trimmed.append(pad_value)

        seqs[i] = s_trimmed
        
    return np.array(seqs, dtype=int)
    pass