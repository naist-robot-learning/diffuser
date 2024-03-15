import json
import numpy as np
from os.path import join
import ipdb
import time

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.sampling.functions import n_step_guided_p_sample
import sys
sys.path.insert(0, '/home/ws/src')
from CoppeliaEnv4Diffuser.gymEnvironments import CoppeliaGym, CoppeliaGymFull
import matplotlib.pyplot as plt
 
class Parser(utils.Parser):
    dataset: str = 'ur5_coppeliasim_full_path_plus_hand_v1'
    config: str = 'config.robo'

#---------------------------------- setup ----------------#import ipdb; ipdb.set_trace()------------------#

args = Parser().parse_args('plan')

#logger = utils.Logger(args)
print(" Loading environment... please run CoppeliaSim if not yet running")
env = CoppeliaGymFull()


#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, 
    epoch=args.diffusion_epoch
)
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

################################### Guiding ###################################
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim
print("action_dim: ", action_dim)

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
guide_config = utils.Config(args.guide, model=model, verbose=False)
guide = guide_config()
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

###############################################################################
policy = policy_config()
#policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#


for i in range(0, 30):
    observation = env.reset()
    ## observations for rendering
    rollout = [observation.copy()]
    cond = {}
    total_reward = 0
    for t in range(0,32):
        print("t: ", t)
        state = observation.copy()
    
        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:
            cond[0] = observation
            print("observation: ",  observation)
            #import ipdb; ipdb.set_trace()
            #action, samples = policy(cond, batch_size=args.batch_size)
            action, samples = policy(cond, batch_size=args.batch_size, verbose=args.verbose)
            actions = samples.actions[0]
            sequence = samples.observations[0]
            fullpath = join(args.savepath, f'{t}.png')
            # Create a plot of the actions over time
            plt.plot(actions)
            plt.show()
            
            #renderer.composite(fullpath, samples.observations, ncol=1)
        #import ipdb; ipdb.set_trace()
        next_observation, reward, terminal, _ = env.step(actions[t,:])
        time.sleep(0.005)
        total_reward = reward
        #score = env.get_normalized_score(total_reward)
        score = total_reward
        # print(
        #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        #     f'{action}'
        # )
        
        ## update rollout observations
        rollout.append(next_observation.copy())
    
        # logger.log(score=score, step=t)
    
        if t % args.vis_freq == 0 or terminal:
            fullpath = join(args.savepath, f'{t}.png')
    
            #if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)
    
    
            # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)
    
            ## save rollout thus far
            #renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)
    
            # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)
    
            # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)
    
        if terminal:
            break
    
        observation = next_observation

# logger.finish(t, env.max_episode_steps, score=score, value=0)

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
