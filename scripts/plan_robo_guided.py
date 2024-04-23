import json
import numpy as np
from os.path import join, exists
import time
import torch

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.sampling.functions import n_step_guided_p_sample
import sys

sys.path.insert(0, "/home/ws/src")
from CoppeliaEnv4Diffuser.gymEnvironments import CoppeliaGym, CoppeliaGymFull
import matplotlib.pyplot as plt
import diffuser.robot.UR5kinematicsAndDynamics_vectorized as ur5


class Parser(utils.Parser):
    dataset: str = "ur5_coppeliasim_full_path_goal"
    config: str = "config.robo"


# ---------------------------------- setup ----------------------------------#

args = Parser().parse_args("plan")

# logger = utils.Logger(args)
print(" Loading environment... please run CoppeliaSim if not yet running ")
env = CoppeliaGymFull()


# ---------------------------------- loading ----------------------------------#
diffusion_loadpath = f"diffusion/H48_T20_{args.dataset}"
diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, diffusion_loadpath, epoch=args.diffusion_epoch
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
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon,
    batch_size=args.batch_size,
    action_dim=action_dim,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    normalizer=dataset.normalizer,
    # dim_mults=args.dim_mults,
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
    horizon=args.horizon,
    sample_fn=n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)
print("args.horizon: ", args.horizon)

###############################################################################
policy = policy_config()
# policy = Policy(diffusion, dataset.normalizer)

# ---------------------------------- main loop ----------------------------------#
failure_count = 0
try_replan = False
joint_position_l = []
robot_hand_pose_l = []
goal_pose_l = []
values_l = []
for i in range(0, 100):
    remove_str = "ur5_coppeliasim_full_"
    state_type = args.dataset[len(remove_str) :]
    observation, goal_pose, hand_pose = env.reset(state_type=state_type)
    ## observations for rendering
    rollout = [observation.copy()]
    cond = {}
    total_reward = 0
    if i % 10 == 0 and i != 0:
        print(f"Failed to reach goal pose {failure_count} times so far")

    for t in range(0, args.horizon):
        # print("t: ", t)
        state = observation.copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0 or try_replan:
            print(f"Planning experiment #{i}")
            cond[0] = torch.tensor(observation).to("cuda")
            cond["hand_pose"] = torch.tensor(hand_pose).to("cuda")
            cond["goal_pose"] = torch.tensor(goal_pose).to("cuda")
            # print("observation: ", observation)
            # import ipdb; ipdb.set_trace()
            # action, samples = policy(cond, batch_size=args.batch_size)
            # action, samples = policy(cond, batch_size=args.batch_size, verbose=args.verbose)
            samples = policy(
                cond,
                batch_size=args.batch_size,
                verbose=args.verbose,
            )
            # actions = samples.actions[0]
            if args.scale == 0:
                randint = np.random.randint(64, size=1)[0]
                sequence = utils.to_np(samples.observations[randint])
                value = utils.to_np(samples.values[randint])
            else:
                sequence = utils.to_np(samples.observations[0])
                value = utils.to_np(samples.values[0])
            last_sequence = utils.to_np(samples.observations[-1])

            fullpath = join(args.savepath, f"{t}.png")
            # Create a plot of the actions over time
            # plt.figure(1)
            # plt.plot(sequence[:, 0], label="q0")
            # plt.plot(sequence[:, 1], label="q1")
            # plt.plot(sequence[:, 2], label="q2")
            # plt.plot(sequence[:, 3], label="q3")
            # plt.plot(sequence[:, 4], label="q4")
            # plt.plot(sequence[:, 5], label="q5")
            # Check if sequence reaches goal pose
            goal_pose = torch.tensor(sequence[-1, 6:9]).to("cuda")
            last_q = torch.tensor(sequence[-1, 0:6]).unsqueeze(1).unsqueeze(2)
            last_pose = ur5.fkine(last_q.to("cuda"))[..., :3]
            d = torch.linalg.norm(goal_pose - last_pose)

            # plt.xlabel("Time step")
            # plt.ylabel("Joint angle")
            # plt.title("Diffuser output")
            # plt.legend()
            # # plt.show()

            # plt.figure(2)
            # plt.plot(last_sequence[:, 0], label="q0")
            # plt.plot(last_sequence[:, 1], label="q1")
            # plt.plot(last_sequence[:, 2], label="q2")
            # plt.plot(last_sequence[:, 3], label="q3")
            # plt.plot(last_sequence[:, 4], label="q4")
            # plt.plot(last_sequence[:, 5], label="q5")
            # plt.xlabel("Time step")
            # plt.ylabel("Joint angle")
            # plt.title("Diffuser output")
            # plt.legend()
            # plt.show()

            if d > 0.1:
                try_replan = True
                t = -1
                failure_count += 1
                # Create a plot of the actions over time
                plt.figure(i)
                plt.plot(sequence[:, 0], label="q0")
                plt.plot(sequence[:, 1], label="q1")
                plt.plot(sequence[:, 2], label="q2")
                plt.plot(sequence[:, 3], label="q3")
                plt.plot(sequence[:, 4], label="q4")
                plt.plot(sequence[:, 5], label="q5")

                plt.xlabel("Time step")
                plt.ylabel("Joint angle")
                plt.title("Diffuser output")
                plt.legend()
                plt.savefig(f"failure{i}.png")
                continue
            else:
                try_replan = False

        # renderer.composite(fullpath, samples.observations, ncol=1)
        # import ipdb; ipdb.set_trace()
        # next_observation, reward, terminal, _ = env.step(actions[t, :])
        cond["hand_pose"] = utils.to_np(cond["hand_pose"])
        next_observation, reward, terminal, _ = env.step(sequence[t, 0:6])
        # next_observation, reward, terminal, _ = env.step(sequence[t,0:6])
        time.sleep(0.01) #0.004 250Hz
        total_reward = reward
        # score = env.get_normalized_score(total_reward)
        score = total_reward
        # print(
        #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        #     f'{action}'
        # )

        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        if t % args.vis_freq == 0 or terminal:
            fullpath = join(args.savepath, f"{t}.png")

            # if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)

            # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

            ## save rollout thus far
            # renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

            # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

            # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation
    joint_position_l.append([sequence[:,0:6]])
    robot_hand_pose_l.append([cond["hand_pose"]])
    goal_pose_l.append([sequence[:, 6:13]])
    values_l.append([value])

# logger.finish(t, env.max_episode_steps, score=score, value=0)
index = 1
directory = "logs"
while True:
    filename = join(args.savepath, f"Trial_{index}_{args.scale}.npz")
    if not exists(filename):
        np.savez(
            filename,
            joint_position=joint_position_l,
            robot_hand_pose=robot_hand_pose_l,
            goal_pose=goal_pose_l,
            values=values_l,
        )
        print(f"Arrays saved to: {filename}")
        break
    else:
        index += 1

## save result as a json file
json_path = join(args.savepath, "rollout.json")
json_data = {
    "score": score,
    "step": t,
    "return": total_reward,
    "term": terminal,
    "epoch_diffusion": diffusion_experiment.epoch,
}
json.dump(json_data, open(json_path, "w"), indent=2, sort_keys=True)
