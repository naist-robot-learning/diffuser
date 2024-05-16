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

        # print("n_guide_steps: ", n_guide_steps)
        # print("antes de grad: ", t)
        # start = time.time()
        x, y, y_measured = guide_gradient_steps(
            x,
            cond=cond,
            t=t,
            guide=guide,
            t_stopgrad=t_stopgrad,
            scale=scale,
            action_dim=model.action_dim,
        )
        # end = time.time()
        # print("time to compute gradient: ", end-start)
        # print("despues de grad: ", t)
        # import ipdb; ipdb.set_trace()
        # if scale_grad_by_std:
        #    grad = model_std * grad

        # grad[t < t_stopgrad] = 0
        # nonzero_mask = grad !=0
        # print("gradient ind: ", np.nonzero(nonzero_mask))
        # print("nonzero values: ", grad[nonzero_mask])
        # print("t: ", t)
        # print("x[0,0,:]: ", x[0,0,:])
        # print("grad[0,0,:]: ", grad[0,0,:])
        # x = x + scale * grad

        # print("x[0,0,:]: ", x[0,0,:])
        # x = apply_conditioning(x, cond, model.action_dim)
        # print("x.shape: ", np.shape(x))

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y, y_measured


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
            y, grad_scaled, y_measured = guide.gradients(x, cond, t)

        if scale_grad_by_std:
            grad_scaled = model_var * grad_scaled

        grad_scaled[t < t_stopgrad] = 0
        x = x + scale * grad_scaled
        x = apply_conditioning(x, cond, action_dim)
    return x, y, y_measured
