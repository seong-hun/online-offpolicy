import numpy as np

from fym.core import BaseEnv, BaseSystem
import fym.logging as logging

import config
import figure


def get_q(x):
    return x.T.dot(config.Q).dot(x)


def get_Phia(x):
    x1, x2 = x
    co = np.cos(2 * x1) + 2
    return np.vstack((x1 * co, 2 * x2 * co)) / 2


def get_phic(x):
    x1, x2 = x
    return np.vstack((x1**2, x1 * x2, x2**2))


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
    def set_dot(self, u):
        self.dot = - (self.state - u) / config.TAUF


class Filter(BaseEnv):
    def __init__(self, phic0, na):
        super().__init__()
        self.qf = LPF()
        self.phicf = LPF(phic0)
        self.prpf = LPF(shape=(na, na))
        self.pruf = LPF(shape=(na, 1))

    def set_dot(self, x, u):
        R, Phia = config.R, get_Phia(x)
        q = get_q(x)
        phic = get_phic(x)
        prp = Phia.dot(R).dot(Phia.T)
        pru = Phia.dot(R).dot(u)

        self.qf.set_dot(q)
        self.phicf.set_dot(phic)
        self.prpf.set_dot(prp)
        self.pruf.set_dot(pru)


class MemorySystem(BaseSystem):
    def set_dot(self, u):
        self.dot = - config.K * self.state + u


class Memory(BaseEnv):
    def __init__(self, nc, na):
        super().__init__()
        N = nc + na
        self.Y = MemorySystem(shape=(N, 1))
        self.Xi = MemorySystem(shape=(N, N))
        self.D = MemorySystem(shape=(N, 1))
        self.Dast = MemorySystem(shape=(N, 1))

    def set_dot(self, xi, y, d, dast):
        self.Y.set_dot(xi.dot(y))
        self.Xi.set_dot(xi.dot(xi.T))
        self.D.set_dot(xi.dot(d))
        self.Dast.set_dot(xi.dot(dast))


class ActorCritic(BaseEnv):
    def __init__(self, nc, na):
        super().__init__()
        self.wc = BaseSystem(shape=(nc, 1))
        self.wt = BaseSystem(shape=(na, 1))
        self.wa = BaseSystem(shape=(na, 1))

    def set_dot(self, x, filter_state):
        ac_state = self.observe_list()
        qf, phicf, prpf, pruf = filter_state
        wc, wt, wa = ac_state

        phic = get_phic(x)
        dphicf = 1 / config.TAUF * (phic - phicf)
        EPS = 1e-8
        e = self.get_error(x, filter_state, ac_state)
        e = e / (np.abs(e) + EPS)
        self.wc.dot = - config.ETA * dphicf * e
        self.wt.dot = - config.ETA * (-2) * (prpf.dot(wa) - pruf) * e
        self.wa.dot = - 1 * (wa - wt)

    def get_error(self, x, filter_state, ac_state):
        qf, phicf, prpf, pruf = filter_state
        wc, wt, wa = ac_state

        phic = get_phic(x)
        dphicf = 1 / config.TAUF * (phic - phicf)
        e = (
            wc.T.dot(dphicf)
            - 2 * wt.T.dot(prpf).dot(wa)
            + 2 * wt.T.dot(pruf)
            + qf
            + wa.T.dot(prpf).dot(wa)
        )
        return e


class TargetActor(BaseSystem):
    def __init__(self, na):
        super().__init__(shape=(na, 1))

    def set_dot(self, caw):
        self.dot = config.ETA_PI * (self.state - caw)


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=config.TIME_STEP, max_t=config.FINAL_TIME)
        x = config.INITIAL_STATE
        phic = get_phic(x)
        nc = phic.shape[0]
        na = get_Phia(x).shape[0]
        self.system = MainSystem(x)
        self.filter = Filter(phic0=phic, na=na)
        self.actor_critic = ActorCritic(nc, na)

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        state = self.observe_dict(y)
        x = state["system"]
        filter_state = state["filter"].values()
        ac_state = state["actor_critic"].values()

        u = self.get_behavior(t, x)

        R, Phia = config.R, get_Phia(x)
        q = get_q(x)
        phic = get_phic(x)
        prp = Phia.dot(R).dot(Phia.T)
        pru = Phia.dot(R).dot(u)

        e = self.actor_critic.get_error(x, filter_state, ac_state)
        true_ac_state = config.WCAST, config.WAAST, config.WAAST
        true_e = self.actor_critic.get_error(x, filter_state, true_ac_state)
        return dict(
            time=t,
            state=state,
            control=u,
            filter_true=dict(q=q, phic=phic, prp=prp, pru=pru),
            e=e,
            true_e=true_e,
        )

    def step(self):
        *_, done = self.update()
        return done

    def set_dot(self, t):
        x = self.system.state
        filter_state = self.filter.observe_list()

        # Main system
        u = self.get_behavior(t, x)
        self.system.set_dot(u)

        # Filter dynamics
        self.filter.set_dot(x, u)

        # Actor-critic dynamics
        self.actor_critic.set_dot(x, filter_state)

    def get_behavior(self, t, x):
        x1, x2 = x
        u = - x2[:, None]
        u += 0.3 * np.sin(1/20 * t) * np.cos(4 * np.exp(-t/30) * t + 1)
        u += 0.2 * np.exp(-t/20) * np.sin(t + 2)
        return u


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
