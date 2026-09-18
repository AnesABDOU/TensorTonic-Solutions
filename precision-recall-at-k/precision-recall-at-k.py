def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    if k <= 0.0:
        return [0.0, 0.0]
        
    top_k = recommended[:k]
    matchs = 0
    for i in range(len(relevant)):
        if relevant[i] in top_k:
            matchs += 1

    precision = matchs/k
    recall = matchs/len(relevant) if len(relevant) != 0 else None
    return [precision, recall]
    
    pass