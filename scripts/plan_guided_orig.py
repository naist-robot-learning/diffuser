import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import matplotlib.pyplot as plt
import csv


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
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
#guide_config = utils.Config(args.guide, model=model, verbose=False)
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

env = dataset.env
observation = env.reset()

## observations for rendering
rollout = [observation.copy()]
r_t = []
x_s = []
t_s = []
total_reward = 0
for t in range(args.max_episode_length+1):

    if t % 10 == 0: print(args.savepath, flush=True)

    ## save state for rendering only
    state = env.state_vector().copy()
    #print("env observation space: ", env.observation_space)
    ## format current observation for conditioning
    conditions = {0: observation}
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

    ## execute action in environment
    next_observation, reward, terminal, info = env.step(action)

    ## print reward and score
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'values: {samples.values} | scale: {args.scale}',
        flush=True,
    )

    ## update rollout observations
    rollout.append(next_observation.copy())
    r_t.append(total_reward)
    x_s.append(info["reward_run"].copy())
    t_s.append(t)
    ## render every `args.vis_freq` steps
    logger.log(t, samples, state, rollout)

    if terminal:
        break

    observation = next_observation
    if t % 200 == 0:
        with open("exp_costfn_05ms.csv", 'w', newline='') as csv_file:
            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)
            data_dict = {
                 'reward_run': x_s,
                 'total_reward': r_t,
                 'time_step':t_s,
            }
            # Write the list of tuples to the CSV file
            csv_writer.writerow(data_dict.keys())  # Write header
    
            # Write the data row by row
            for row in zip(*data_dict.values()):
                csv_writer.writerow(row)
                        
## write results to json file at `args.savepath`
logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
