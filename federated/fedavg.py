
def fedavg(weights):
    avg_weights = {}
    for key in weights[0].keys():
        avg_weights[key] = sum([w[key] for w in weights]) / len(weights)
    return avg_weights
