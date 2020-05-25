import numpy as np
import matplotlib.pyplot as plt

import fym.logging as logging

import main


def plot():
    data = logging.load("data/tmp.h5")
    time = data["time"]
    state = data["state"]

    plt.subplot(221)
    plt.plot(time, state["system"].squeeze())
    plt.legend([r"$x_1$", r"$x_2$"])

    plt.subplot(222)
    Y = state["memory"]["Y"]
    Xi = state["memory"]["Xi"]
    Dast = state["memory"]["Dast"]
    wast = main.Env.wast
    X = np.einsum("bij,jk->bik", Xi, wast) - Dast
    # plt.plot(time, state["filter"]["dast"].squeeze(), label=r"$d^\ast$")
    plt.plot(time, Y.squeeze(), "k", label=r"$Y$")
    plt.plot(time, X.squeeze(), "k--", label=r"$\Xi w^\ast - D^\ast$")
    plt.legend()

    plt.subplot(223)
    plt.plot(time, state["actor_critic"]["wa"].squeeze())

    plt.subplot(224)
    plt.plot(time, state["actor_critic"]["wc"].squeeze())

    plt.show()


if __name__ == "__main__":
    plot()
