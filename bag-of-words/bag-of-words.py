import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    # Write code here
    bow = []
    token_dict = {}
    for token in tokens:
        token_dict[token] = token_dict.get(token, 0) + 1

    for word in vocab:
        if word in token_dict.keys():
            bow.append(token_dict[word])
        else:
            bow.append(0)
    return np.asarray(bow)
    