from collections import namedtuple
import torch
import einops

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn
from diffusionRenderer import DiffusionAnimator

Trajectories = namedtuple("Trajectories", "observations values values_measured")


class GuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        trajectories = samples.trajectories

        normed_observations = trajectories[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(normed_observations, "observations")

        trajectories = Trajectories(observations, samples.values, samples.values_measured)
        diffusion_chain = samples.chains

        render = False
        if render:

            renderer = DiffusionAnimator()
            diffusion_chain = self.normalizer.unnormalize(diffusion_chain, "observations")

            for i in range(20):

                x = diffusion_chain[:64, i]
                renderer.load_trajectory(x)
                renderer.render_robot_animation(i)
            x = trajectories.observations[0].unsqueeze(0)
            renderer.load_trajectory(x)
            renderer.render_robot_animation(20, True)
        return trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        # Remove hand_pose for normalization
        condition_obs = dict([next(iter(conditions.items()))])
        condition_obs = utils.apply_dict(
            self.normalizer.normalize,
            condition_obs,
            "observations",
        )
        # Concatenate back with hand_pose
        key_obs = next(iter(condition_obs.keys()))
        conditions[key_obs] = condition_obs[key_obs]
        conditions = utils.to_torch(conditions, dtype=torch.float32, device="cuda:0")
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            "d -> repeat d",
            repeat=batch_size,
        )
        return conditions
