import numpy as np
import matplotlib.pyplot as plt

import fym.logging as logging

import main
import config


def plot():
    data = logging.load("data/tmp.h5")
    time = data["time"]
    state = data["state"]

    plt.figure()
    ax = plt.subplot(211, title="State")
    plt.plot(time, state["system"].squeeze())
    plt.legend([r"$x_1$", r"$x_2$"], loc="best")

    plt.subplot(212, title="Control", sharex=ax)
    plt.plot(time, data["control"].squeeze())
    plt.xlabel("Time [s]")

    plt.tight_layout()

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

    ax1 = plt.subplot(221, title="Critic")
    plt.plot(time, state["actor_critic"]["wc"].squeeze(), "k")
    plt.hlines(config.WCAST.squeeze(), 0, time.max(), colors="r", linestyles="--")
    plt.ylim(-0.5, 2)
    plt.xlim(0, time.max())

    ax2 = plt.subplot(222, title="Actor")
    plt.plot(time, state["actor_critic"]["wa"].squeeze(), "k")
    plt.hlines(config.WAAST.squeeze(), 0, time.max(), colors="r", linestyles="--")
    # plt.ylim(-3, 0)
    plt.xlim(0, time.max())

    plt.subplot(223, title="Actor Target", sharex=ax1)
    plt.plot(time, state["actor_critic"]["wt"].squeeze(), "k")
    plt.hlines(config.WAAST.squeeze(), 0, time.max(), colors="r", linestyles="--")
    # plt.ylim(-3, 0)
    plt.xlim(0, time.max())
    plt.xlabel("Time [s]")

    plt.subplot(224, title="HJB Error", sharex=ax2)
    plt.plot(time, data["e"].squeeze(), "k")
    plt.plot(time, data["true_e"].squeeze(), "r--")
    # plt.ylim(-0.2, 0.2)
    plt.xlim(0, time.max())
    plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot()
