import torch
import numpy as np
import time

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)


@torch.no_grad()
def n_step_guided_p_sample(
    model,
    x,
    cond,
    t,
    guide,
    scale=0.001,
    t_stopgrad=0,
    n_guide_steps=1,
    scale_grad_by_std=True,
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)

    for _ in range(n_guide_steps):

        x, cost_dict, cost_measured = guide_gradient_steps(
            x,
            cond=cond,
            t=t,
            guide=guide,
            t_stopgrad=t_stopgrad,
            scale=scale,
            action_dim=model.action_dim,
        )
        
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, cost_dict, cost_measured


def guide_gradient_steps(
    x,
    cond,
    t,
    guide=None,
    n_guide_steps=3,
    scale_grad_by_std=False,
    model_var=None,
    t_stopgrad=None,
    scale=1.0,
    action_dim=None,
    debug=False,
    **kwargs
):
    for _ in range(n_guide_steps):
        with torch.enable_grad():
            cost_dict, grad_scaled, cost_measured = guide.gradients(x, cond, t)

        if scale_grad_by_std:
            grad_scaled = model_var * grad_scaled

        grad_scaled[t < t_stopgrad] = 0
        x = x + grad_scaled
        x = apply_conditioning(x, cond, action_dim)
    return x, cost_dict, cost_measured
