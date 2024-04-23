import socket

from diffuser.utils import watch

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("dataset", ""),
]


plan_args_to_watch = [
    ("prefix", ""),
    ##
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("value_horizon", "V"),
    ("discount", "d"),
    ("normalizer", ""),
    ("batch_size", "b"),
    ##
    ("conditional", "cond"),
]

base = {
    "diffusion": {
        ## model
        "model": "models.TemporalUnet",
        "diffusion": "models.GaussianDiffusion",
        "horizon": 256,
        "n_diffusion_steps": 256,
        "action_weight": 1,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": False,
        "dim_mults": (1, 4, 8),
        "renderer": "",
        ## dataset
        "loader": "datasets.TrajectoryDataset",
        "termination_penalty": None,
        "normalizer": "GaussianNormalizer",
        "preprocess_fns": [],  # ['maze2d_set_terminals'],
        "clip_denoised": False,  # MAze is true
        "use_padding": True,  # Maze is false
        "max_path_length": 49,
        ## serialization
        "logbase": "logs",
        "prefix": "diffusion/",
        "exp_name": watch(diffusion_args_to_watch),
        ## training
        "n_steps_per_epoch": 1000,
        "loss_type": "l2",
        "n_train_steps": 2e6,
        "batch_size": 64,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.995,
        "save_freq": 1000,
        "sample_freq": 1000,
        "n_saves": 100,
        "save_parallel": False,
        "n_reference": 50,
        "n_samples": 10,
        "bucket": None,
        "device": "cuda",
        "steps_til_summary": 1000,
    },
    "plan": {
        "model": "models.cost_function.CostFn",  # New Cost Fn !
        "guide": "sampling.guides.ValueGuide",
        "policy": "sampling.policies.GuidedPolicy",
        "batch_size": 64,
        "preprocess_fns": [],
        "device": "cuda",
        ## sample_kwargs
        "n_guide_steps": 2,
        "scale": 0.1,
        "t_stopgrad": 2,
        "scale_grad_by_std": True,
        ## diffusion model
        "horizon": 256,
        "n_diffusion_steps": 256,
        "normalizer": "LimitsNormalizer",
        ## serialization
        "vis_freq": 10,
        "logbase": "logs",
        "prefix": "plans/release",
        "exp_name": watch(plan_args_to_watch),
        "suffix": "0",
        "conditional": False,
        ## loading
        "diffusion_loadpath": "f:diffusion/H{horizon}_T{n_diffusion_steps}_{dataset}",
        "diffusion_epoch": "latest",
        "verbose": True,
    },
}

# ------------------------ overrides ------------------------#

"""
    robo episode steps:
        small: 150
        medium: 250
        large: 600
"""

ur5_coppeliasim_full_path_goal = {
    "diffusion": {
        "horizon": 48,  # longest path in dataset
        "n_diffusion_steps": 20,
        "attention": True,
        "use_actions": False,
    },
    "plan": {
        "horizon": 48,
        "n_diffusion_steps": 20,
        "use_actions": False,
        "scale": 0.1,
        "t_stopgrad": 0,
    },
}
ur5_coppeliasim_full_path_goal_hand = {
    "diffusion": {
        "horizon": 48,  # longest path in dataset
        "n_diffusion_steps": 16,
        "attention": True,
        "use_actions": False,
    },
    "plan": {
        "horizon": 48,
        "n_diffusion_steps": 16,
        "use_actions": False,
        "scale": 0.00000000000001,
        "t_stopgrad": 2,
    },
}
ur5_coppeliasim_full_path_goal_hand_no_orientation = {
    "diffusion": {
        "horizon": 48,  # longest path in dataset
        "n_diffusion_steps": 16,
        "attention": True,
        "use_actions": False,
    },
    "plan": {
        "horizon": 48,
        "n_diffusion_steps": 16,
        "use_actions": False,
        "scale": 0.00000000000001,
        "t_stopgrad": 2,
    },
}
