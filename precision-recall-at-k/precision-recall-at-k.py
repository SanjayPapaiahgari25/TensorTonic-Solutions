def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    n = len(relevant)
    recommended, relevant = set(recommended[:k]), set(relevant)

    intersection = recommended.intersection(relevant)

    precision_at_k = float(len(intersection)/k)
    recall_at_k = float(len(intersection)/n)

    return [precision_at_k, recall_at_k]