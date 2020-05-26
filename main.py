import numpy as np

from fym.core import BaseEnv, BaseSystem
import fym.logging as logging

import config
import figure


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


class LPF(BaseSystem):
    def __init__(self, initial_state=None):
        super().__init__(initial_state)
        # TODO

    def set_dot(self, u):
        self.dot = - (self.state - u) / config.TAUF


class Filter(BaseEnv):
    def __init__(self, phic0, na):
        super().__init__()
        self.y = BaseSystem()
        self.xic = BaseSystem(phic0)
        self.xia = BaseSystem(shape=(na, 1))
        self.d = BaseSystem()
        self.dast = BaseSystem()

        self.y = LPF()
        self.xic = LPF(phic0)
        self.xia = LPF(shape=(na, 1))
        self.d = LPF()
        self.dast = LPF()

    def set_dot(self, q, phic, pru, uhat_square, uast_square):
        self.y.dot = -(self.y.state + q) / config.TAUF
        self.xic.dot = -(self.xic.state - phic) / config.TAUF
        self.xia.dot = -(self.xia.state - 2 * pru) / config.TAUF
        self.d.dot = -(self.d.state - uhat_square) / config.TAUF
        self.dast.dot = -(self.dast.state - uast_square) / config.TAUF


class Memory(BaseEnv):
    def __init__(self, nc, na):
        super().__init__()
        N = nc + na
        self.Y = BaseSystem(shape=(N, 1))
        self.Xi = BaseSystem(shape=(N, N))
        self.D = BaseSystem(shape=(N, 1))
        self.Dast = BaseSystem(shape=(N, 1))

    def set_dot(self, xi, y, d, dast):
        self.Y.dot = -config.K * self.Y.state + xi.dot(y)
        self.Xi.dot = -config.K * self.Xi.state + xi.dot(xi.T)
        self.D.dot = -config.K * self.D.state + xi.dot(d)
        self.Dast.dot = -config.K * self.Dast.state + xi.dot(dast)


class ActorCritic(BaseEnv):
    def __init__(self, nc, na):
        super().__init__()
        self.wc = BaseSystem(shape=(nc, 1))
        self.wa = BaseSystem(shape=(na, 1))

    def set_dot(self, Xi, Y, D, cprpc, d):
        w = self.state[:, None]

        self.dot = (
            - config.ETA * (Xi.dot(w) - Y - D)
            - 1 / config.TAUF * cprpc.dot(w) * d
        )


class Env(BaseEnv):
    Q = np.eye(2)
    R = 1
    wast = np.vstack((0.5, 0, 1, 0.5, 1))

    def __init__(self):
        super().__init__(dt=config.TIME_STEP, max_t=config.FINAL_TIME)
        x = config.INITIAL_STATE
        phic = self.phic(x)
        nc = phic.shape[0]
        na = self.Phia(x).shape[0]
        self.system = MainSystem(x)
        self.actor_critic = ActorCritic(nc, na)
        self.filter = Filter(phic0=phic, na=na)
        self.memory = Memory(nc, na)

        self.Ca = np.block([np.zeros((na, nc)), np.eye(na)])

    def step(self):
        *_, done = self.update()
        return done

    def set_dot(self, t):
        x, w, (y, xic, xia, d, dast), (Y, Xi, D, Dast) = self.observe_list()
        w = np.vstack(w)

        # print(np.linalg.eigvals(Xi).min())

        # Main system
        x1, x2 = x
        u = - x2[:, None] + 0.1 * np.sin(2 * t) + 0.15 * np.cos(4 * t + 1)
        self.system.set_dot(u)

        # Parameter dynamics
        Ca, R = self.Ca, config.R
        Phia = self.Phia(x)

        cprpc = Ca.T.dot(Phia).dot(R).dot(Phia.T).dot(Ca)
        self.actor_critic.set_dot(Xi, Y, D, cprpc, d)

        # Filter dynamics
        q = self.q(x)
        phic = self.phic(x)
        pru = Phia.dot(R).dot(u)
        uhat = Phia.T.dot(Ca).dot(w)
        uast = Phia.T.dot(Ca).dot(self.wast)
        uhat_square = uhat.T.dot(R).dot(uhat)
        uast_square = uast.T.dot(R).dot(uast)

        self.filter.set_dot(q, phic, pru, uhat_square, uast_square)

        # Memory dynamics
        xi = np.vstack((
            1 / config.TAUF * (phic - xic),
            xia
        ))
        self.memory.set_dot(xi, y, d, dast)

    def q(self, x):
        return x.T.dot(config.Q).dot(x)

    def Phia(self, x):
        x1, x2 = x
        co = np.cos(2 * x1) + 2
        return np.vstack((x1 * co, 2 * x2 * co)) / 2

    def phic(self, x):
        x1, x2 = x
        return np.vstack((x1**2, x1 * x2, x2**2))

    def phia(self, x):
        pass


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

    figure.plot()


if __name__ == "__main__":
    main()
