"""The flow matching maths, shared by any expert that denoises actions.

Training. Mix a clean action chunk with noise at a time t, and ask the model to predict
the velocity that points from the action towards the noise:

    x_t = t * noise + (1 - t) * action      t = 0 -> the clean action, t = 1 -> pure noise
    u   = noise - action                    the velocity the model has to predict

Inference. Start from pure noise at t = 1 and integrate the predicted velocity back down
to t = 0 with Euler steps, which lands on an action chunk.

This convention - which end of the interval is clean - is the one openpi, lerobot and
SmolVLA all use. Reversing it trains a model that confidently produces noise, and nothing
about the code will look wrong, so it is worth stating once here.
"""

import math

import torch

# SmolVLA samples the training time as Beta(1.5, 1.0) scaled into this range.
TRAIN_TIME_ALPHA = 1.5
TRAIN_TIME_BETA = 1.0
TRAIN_TIME_SCALE = 0.999
TRAIN_TIME_OFFSET = 0.001


def sinusoidal_time_embedding(time, dimension, max_period=10000.0):
    """Turn a batch of times in [0, 1] into (batch, dimension) sine and cosine features.

    time: (batch,) float. `dimension` must be even: half of it is sine, half cosine.
    """
    half = dimension // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=time.device) / half)
    angles = time[:, None] * frequencies[None, :]                  # (batch, half)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def sample_training_time(batch_size, device):
    """Draw one training time per sample, as SmolVLA does.

    Beta(1.5, 1.0) has density proportional to sqrt(t), so more samples land near t = 1
    (the noisy end) than near t = 0. The scale and offset keep t away from exactly 0 and
    1, where the target velocity is degenerate.
    """
    beta = torch.distributions.Beta(torch.tensor(TRAIN_TIME_ALPHA),
                                    torch.tensor(TRAIN_TIME_BETA))
    time = beta.sample((batch_size,)) * TRAIN_TIME_SCALE + TRAIN_TIME_OFFSET
    return time.to(device)


def make_noisy_actions(action, noise, time):
    """x_t for the given times.

    action, noise: (batch, chunk, action_dim). time: (batch,), broadcast over the last
    two axes.
    """
    time = time[:, None, None]
    return time * noise + (1 - time) * action


def velocity_target(action, noise):
    """The velocity the model has to predict: from the clean action towards the noise."""
    return noise - action


def sample_euler(denoise, noise, num_steps):
    """Integrate the predicted velocity from t = 1 (pure noise) down to t = 0 (actions).

    denoise: callable(x_t, time) -> velocity, same shape as x_t, with time of shape
             (batch,). Each Euler step asks it once.
    noise:   (batch, chunk, action_dim), the sample to start from.

    The step is negative because time runs backwards: 1.0, 0.9, ... , 1 / num_steps.
    """
    step_size = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * step_size
        time_tensor = torch.full((noise.shape[0],), time, device=noise.device)
        velocity = denoise(x_t, time_tensor)
        x_t = x_t + step_size * velocity
    return x_t
