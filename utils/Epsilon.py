class EpsilonScheduler:
    def __init__(self, start: float, end: float, decay_steps: int):
        self.start = start
        self.end = end
        self.decay_steps = max(1, decay_steps)

    def value(self, t: int) -> float:
        frac = min(1.0, t / self.decay_steps)
        return self.start + (self.end - self.start) * frac


def hard_update(target, online):
    target.load_state_dict(online.state_dict())


def soft_update(target, online, tau: float):
    for t_param, o_param in zip(target.parameters(), online.parameters()):
        t_param.data.mul_(1.0 - tau).add_(tau * o_param.data)
