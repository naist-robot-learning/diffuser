import torch
import numpy as np

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)


@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            #print("n_guide_steps: ", n_guide_steps)
            #print("antes de grad: ", t)
            y, grad = guide.gradients(x, cond, t)
            #print("despues de grad: ", t)

        if scale_grad_by_std:
            grad = model_std * grad
         
        grad[t < t_stopgrad] = 0
        #nonzero_mask = grad !=0
        #print("gradient ind: ", np.nonzero(nonzero_mask))
        #print("nonzero values: ", grad[nonzero_mask])
        #print("t: ", t)
        #print("x[0,0,:]: ", x[0,0,:])
        #print("grad[0,0,:]: ", grad[0,0,:])
        x = x + scale * grad
        #print("x[0,0,:]: ", x[0,0,:])
        x = apply_conditioning(x, cond, model.action_dim)
        #print("x.shape: ", np.shape(x))

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y
