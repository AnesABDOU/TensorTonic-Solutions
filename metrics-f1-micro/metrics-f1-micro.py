def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    TP = 0.0
    FN = 0.0
    FP = 0.0
    
    if y_true == y_pred:
        return 1.0
        
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            TP += 1.0
        elif y_true[i] != y_pred[i]:
            FP += 1.0
            FN += 1.0
    return 2*TP / (2*TP + FP + FN)
    pass