"""Exponential moving average of the trained weights.

Keeps a shadow copy of the trainable parameters and folds the live ones into it after
every step. Evaluating the shadow copy instead of the live weights gives steadier numbers,
which matters here because the flow matching loss is random - it re-draws the noise and the
time at every step - so a single epoch is a noisy estimate.

The decay schedule is the one mini-vla uses for its diffusion policy: decay starts near 0
(a straight copy) and climbs towards `max_value`, so the earliest steps follow the live
weights and the average gets stickier over time.

Only the trainable parameters are tracked. The frozen trunk is never updated, so there is
nothing to average there.
"""

import torch


class EMA:
    """A shadow copy of some tensors, folded towards their live values each step."""

    def __init__(self, parameters, power=0.75, max_value=0.9999):
        self.power = power
        self.max_value = max_value
        self.step_count = 0
        self.shadow = []
        for parameter in parameters:
            self.shadow.append(parameter.detach().clone())

    def step(self, parameters):
        """Fold the current weights into the shadow copy."""
        decay = min(1 - (1 + self.step_count) ** -self.power, self.max_value)
        self.step_count += 1
        with torch.no_grad():
            for shadow, parameter in zip(self.shadow, parameters):
                # shadow = decay * shadow + (1 - decay) * live weights
                shadow.mul_(decay)
                shadow.add_(parameter.detach(), alpha=1 - decay)

    def swap_in(self, parameters):
        """Write the shadow weights into the live parameters, returning what was there.

        Used to evaluate the averaged weights without building a second copy of a 300M
        parameter trunk just to hold them.
        """
        saved = []
        with torch.no_grad():
            for shadow, parameter in zip(self.shadow, parameters):
                saved.append(parameter.detach().clone())
                parameter.copy_(shadow)
        return saved

    def swap_out(self, parameters, saved):
        """Put back the weights that swap_in replaced."""
        with torch.no_grad():
            for parameter, value in zip(parameters, saved):
                parameter.copy_(value)


def make_ema(parameters, power=0.75):
    """Build an EMA over the given tensors."""
    return EMA(parameters, power=power)
