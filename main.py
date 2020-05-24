import numpy as np

from fym.core import BaseEnv, BaseSystem
import fym.logging as logging

import config


class MainSystem(BaseSystem):
    """
    Model from (2010 Vamvoudakis & Lewis)
    """
    coeffs = [-1, 1, -0.5, -0.5]

    def __init__(self, init):
        super().__init__(init)

    def set_dot(self, u):
        x1, x2 = self.state
        a, b, c, d = self.coeffs
        x1dot = a * x1 + b * x2
        x2dot = (
            c * x1 + d * x2 * (1 - (np.cos(2 * x1) + 2)**2)
            + (np.cos(2 * x1) + 2) * u
        )
        self.dot = np.vstack((x1dot, x2dot))


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=config.TIME_STEP, max_t=config.FINAL_TIME)
        self.system = MainSystem(config.INITIAL_STATE)

    def step(self):
        *_, done = self.update()
        return done

    def set_dot(self, t):
        u = 0
        self.system.set_dot(u)


def main():
    env = Env()
    env.logger = logging.Logger("data/tmp.h5")

    env.reset()

    while True:
        env.render()
        done = env.step()
        if done:
            break

    env.close()

    import matplotlib.pyplot as plt

    data = logging.load("data/tmp.h5")
    plt.plot(data["time"], data["state"]["system"].squeeze())
    plt.legend([r"$x_1$", r"$x_2$"])
    plt.show()


if __name__ == "__main__":
    main()
