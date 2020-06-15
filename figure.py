import numpy as np
import matplotlib.pyplot as plt

import fym.logging as logging

import main
import config


def plot():
    data = logging.load("data/tmp.h5")
    time = data["time"]
    state = data["state"]

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(time, state["system"].squeeze())
    # plt.legend([r"$x_1$", r"$x_2$"])

    # plt.subplot(212)
    # plt.plot(time, data["control"].squeeze())

    # plt.figure()
    # for i, key in enumerate(data["filter_true"].keys()):
    #     filter_state = state["filter"][key + "f"].squeeze()
    #     filter_state = filter_state.reshape(filter_state.shape[0], -1)
    #     true_state = data["filter_true"][key].squeeze()
    #     true_state = true_state.reshape(true_state.shape[0], -1)

    #     plt.subplot(221 + i)
    #     plt.plot(time, filter_state, "k")
    #     plt.plot(time, true_state, "r--")

    plt.figure()

    plt.subplot(221)
    plt.plot(time, state["actor_critic"]["wc"].squeeze(), "k")
    plt.hlines(config.WCAST.squeeze(), 0, 20, colors="r", linestyles="--")
    plt.ylim(-2, 2)
    plt.xlim(0, time.max())

    plt.subplot(222)
    plt.plot(time, state["actor_critic"]["wa"].squeeze(), "k")
    plt.hlines(config.WAAST.squeeze(), 0, 20, colors="r", linestyles="--")
    plt.ylim(-2, 2)
    plt.xlim(0, time.max())

    plt.subplot(224)
    plt.plot(time, data["e"].squeeze())
    plt.ylim(-2, 2)
    plt.xlim(0, time.max())

    plt.show()


if __name__ == "__main__":
    plot()
