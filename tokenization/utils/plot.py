from typing import List
import matplotlib.pyplot as plt

def plot_distribution(values: List[float], probs: List[float]) -> None:
    """
    Plots distribution of PMI scores.
    :param values: PMI scores
    :param probs: PMI score probabilities
    :return: None
    """
    fig, ax = plt.subplots()
    ax.plot(values, probs)
    ax.set_xlabel("PMI score")
    ax.set_xlim(min(values), max(values))
    ax.set_ylabel("PMI score probability")
    ax.set_ylim(0, max(probs))
    fig.savefig("pmi_distribution.png")
    plt.show()

def plot_cummulative_dsitribution(values: List[float], probs: List[float]) -> None:
    """
    Plots cummulative distribution of PMI scores.
    :param values: PMI scores
    :param probs: PMI score probabilities
    :return: None
    """
    fig, ax = plt.subplots()
    cummulative_probs = [0.]
    for i in range(len(values)):
        cummulative_probs.append(cummulative_probs[-1] + probs[i])
    cummulative_probs = cummulative_probs[1:]
        
    ax.plot(values, cummulative_probs)
    ax.set_xlabel("PMI score")
    ax.set_xlim(min(values), max(values))
    ax.set_ylabel("PMI score cummulative probability")
    ax.set_ylim(min(cummulative_probs), max(cummulative_probs))
    fig.savefig("pmi_cummulative_distribution.png")
    plt.show()