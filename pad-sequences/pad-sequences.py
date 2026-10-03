import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    m = len(seqs)
    if m is 0:
        return np.array([], dtype=np.int64).reshape(0, 0)
    if max_len is None:
        for i in range(m):
            if max_len is None:
                max_len = len(seqs[i])
            else:
                max_len = max(max_len, len(seqs[i]))
    for i in range(m):
        if len(seqs[i]) > max_len:
            seqs[i] = seqs[i][:max_len]
        for j in range(max_len):
            if j<len(seqs[i]):
                continue
            seqs[i].append(pad_value)
    
    return np.array(seqs)[:,:max_len]