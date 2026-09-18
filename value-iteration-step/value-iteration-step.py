import numpy as np

def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    rewards = np.asarray(rewards)
    transitions = np.asarray(transitions)
    values = np.asarray(values)

    res = np.zeros(len(values))
    
    # we need to sum over the line of the multiplication matrix
    for s in range(len(values)):
        action_scores = rewards[s, :] + gamma * np.dot(transitions[s,:,:], values)
        res[s] = np.max(action_scores)
    return res.tolist()
    pass