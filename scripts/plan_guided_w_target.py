import pdb
import pickle

import diffuser.sampling as sampling
import diffuser.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from tools import dict2pickle
from diffuser.environments.half_cheetah import HalfCheetahFullObsEnv

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')


#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
value_experiment = utils.load_diffusion(
     args.loadbase, args.dataset, args.value_loadpath,
     epoch=args.value_epoch, seed=args.seed,
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    #dim_mults=args.dim_mults,
    device=args.device,
)
model = model_config()

## initialize value guide
# value_function = value_experiment.ema
#guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide_config = utils.Config(args.guide, model=model, verbose=False)
guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

logger = logger_config()
policy = policy_config()


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
total_r = [0] * (args.max_episode_length+1)
x_s = [0] * (args.max_episode_length+1)

env =  dataset.env
observation = env.reset()

## observations for rendering
rollout = [observation.copy()] * (args.max_episode_length+1)
################################
with open('target_episode_500.pickle', 'rb') as file:
    target_episode = pickle.load(file)
################################
#print("target_episode: ", target_episode)
target_next_observations = target_episode['next_observations']
target_next_rewards = target_episode['rewards']
print("target_episode keys(): ", target_episode.keys())
target_observations = target_episode['observations']
print("target_next_observations shape: ", np.shape(target_next_observations))
print("target_observations shape: ", np.shape(target_observations))

target_s = [observation[8].copy()] * (args.max_episode_length+1)
target_r = [0] * (args.max_episode_length+1)

total_reward = 0
total_target_reward = 0
for t in range(args.max_episode_length):

    if t % 10 == 0: print(args.savepath, flush=True)

    ## save state for rendering only
    state = env.state_vector().copy()
    target_vec = target_next_observations[t,:]
    ## format current observation for conditioning
    modified_vel = observation.copy()
    print("observation shape: ", np.shape(observation))
    modified_vel[8] = 10 #m/s
    conditions = {
        0: observation,
        3: target_vec
        }
    
    # profiler = cProfile.Profile()
    # profiler.enable()
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
    
    # profiler.disable()
    # profiler.print_stats(sort='cumulative')  # Print the profiling results

    ## execute action in environment
    next_observation, reward, terminal, info = env.step(action)

    ## print reward and score
    total_reward += reward
    total_target_reward += target_next_rewards[t]
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'values: {samples.values} | scale: {args.scale}',
        flush=True,
    )

    ## update rollout observations
    rollout[t] = next_observation.copy()
        
    total_r[t] = total_reward
    target_r[t] = total_target_reward
    
    print("keys:         ", info.keys())
    x_s[t] = info["reward_run"].copy()
    target_s[t+1]=target_vec[8]
    
    ## render every `args.vis_freq` steps
    logger.log(t, samples, state, rollout)

    if terminal:
        break

    observation = next_observation
    if t % 999 == 0:
        data_dict = {
            "total_reward": total_r,
            "rollouts": rollout,
            "x_position": x_s,
            "target_traj": target_s,
            "target_r": target_r
        }
        dict2pickle(data_dict)
        
## write results to json file at `args.savepath`
logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
