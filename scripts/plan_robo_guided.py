import json
import numpy as np
from os.path import join, exists
import time
import torch

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.sampling.functions import n_step_guided_p_sample
import kornia
import sys
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "/home/ws/src")
from CoppeliaEnv4Diffuser.gymEnvironments import CoppeliaGym, CoppeliaGymFull
import matplotlib.pyplot as plt
import diffuser.robot.UR5kinematicsAndDynamics_vectorized as ur5


# Checklist
# - dataset Model
# tomm_mode
# - plan_robo_guided
# - fkine


class Parser(utils.Parser):
    dataset: str = "ur5_coppeliasim_full_path"  # Change in sequence.py also
    config: str = "config.robo"


# ---------------------------------- setup ----------------------------------#

args = Parser().parse_args("plan")

# logger = utils.Logger(args)
print(" Loading environment... please run CoppeliaSim if not yet running ")
env = CoppeliaGymFull()

horizon = 40
# ---------------------------------- loading ----------------------------------#
#diffusion_loadpath = f"diffusion/H48_T25_{args.dataset}"  # _peTrue"
diffusion_loadpath = args.diffusion_loadpath

# dataset_dir = None  # "/home/ws/src/diffuser/logs/ur5_coppeliasim_full_path/"
diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, diffusion_loadpath, epoch=args.diffusion_epoch)
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

weight_grad_cost_smoothness = 1e-6  # mpd 1e-7
weight_grad_cost_reflected_inertia = 1e-3  # -2e-3  # 1e-3
weight_grad_cost_goals = 2e-0  # 1e-3  # without RM 1e-1
weights = [weight_grad_cost_smoothness, weight_grad_cost_reflected_inertia, weight_grad_cost_goals]
########################## Adjust parameters ######################################
tomm_mode = False
plot_joint_traj = False
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
    test_cost=args.test_cost,
    cond_dim=observation_dim,
    normalizer=dataset.normalizer,
    device=args.device,
)
model = model_config()
guide_config = utils.Config(args.guide, model=model, verbose=False, weights=weights)
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

# ---------------------------------- main loop ----------------------------------#
failure_count = 0
try_replan = False
joint_position_l = []
hand_pose_l = []
goal_pose_l = []
values_l = []
computation_time_l = []
trials = 10
for i in range(0, trials):
    remove_str = "ur5_coppeliasim_full_"

    state_type = args.dataset[len(remove_str) :]
    # Only necessary to initialize corner points
    # env.original_env.set_random_start_pose(tomm_mode=tomm_mode)

    # #########TOMM#############
    if tomm_mode:
        ######### Starting position #############
        start_pos_Wt = torch

        # Overwrite corner point
        start_pos_Wt = torch.tensor(
            [0.70307, -0.65419, 1.13277], device="cuda"  # always -0.18 on z
        )  # 1.1239, -0.651284, 0.8644], device="cuda")
        # goal_pos_Wt = torch.tensor([1.0912, -0.56163, 1.0639], device="cuda")
        start_quat_Wt = torch.tensor(
            [
                1.0,  # tilted left 0.96773,  ## no rotation1.0,  # 0.707,  # x
                0.0,  # tilted left -0.0037593,# no rotation # 0.0 -0.707,  # y
                0.00,  # tilted left -0.027087, # no rotation# 0.0, 0.0,  # z
                0.0,  # tilted left-0.25051,  # no rotation 0.0, 0.0,  # w
            ],
            device="cuda",
        )

        copp_start = ur5.tomm_to_coppeliaBase(start_pos_Wt, start_quat_Wt)
        env.original_env.set_start_pose(copp_start.tolist())
        ######### Goal Position
        goal_pos_Wt = torch.tensor(
            [1.1233, -0.35523, 0.9073], device="cuda"  # 65
        )  # 1.1239, -0.651284, 0.8644], device="cuda")
        # goal_pos_Wt = torch.tensor([1.0912, -0.56163, 1.0639], device="cuda")

        ## tilted modified -0.7210394, -0.6710702, -0.1134626, 0.1299737
        goal_quat_Wt = torch.tensor([-0.5630083, -0.5206123, -0.4305665, 0.476022], device="cuda")
        #     [
        #         -0.7210394,  # 0.69181,  # tilted left 0.96773,  ## no rotation1.0,  # 0.707,  # x
        #         -0.6710702,  # 0.64508,  # tilted left -0.0037593,# no rotation # 0.0 -0.707,  # y
        #         -0.1134626,  # ,0.217,  # tilted left -0.027087, # no rotation# 0.0, 0.0,  # z
        #         0.1299737,  # ,-0.24121,  # tilted left-0.25051,  # no rotation 0.0, 0.0,  # w
        #     ],
        #     device="cuda",
        # )

        goal_pose_Wt = torch.cat((goal_pos_Wt, goal_quat_Wt))
        copp_goal = ur5.tomm_to_coppeliaBase(goal_pos_Wt, goal_quat_Wt)
        env.original_env.set_goal_pose(copp_goal.tolist())

    ############ UR5 Normal table configuration #################
    # env.original_env.set_start_pose(np.array([0.15, 0.2, 0.3, 0.7071, 0.7071, 0.0, 0.0]))  # C10
    # env.original_env.set_goal_pose(np.array([0.55, 0.0, 0.1, 0.7071, 0.7071, 0.0, 0.0]))  # C14

    # env.original_env.set_random_goal_pose(xy_only=True, tomm_mode=tomm_mode)
    observation, goal_pose, hand_pose = env.reset(
        state_type=state_type
    )  ##### goal_pose is with respect to the UR5 CoppeliaSim Base
    print("*******observation********\n", observation)
    
    ## observations for rendering
    rollout = [observation.copy()]
    cond = {}
    total_reward = 0
    if i % 10 == 0 and i != 0:
        print(f"Failed to reach goal pose {failure_count} times so far")

    while True:
        replan = False
        for t in range(0, args.horizon):
            
            state = observation.copy()
            print("t: ", t)

            if t == 0:
                computation_time = 0
                start_time = time.time()
                print(f"Planning experiment #{i}")
                cond[0] = torch.tensor(observation).to("cuda")
                # cond[0] = torch.tensor([0.9990606, -0.91376438, -2.03536981, -1.68044293, 1.9546568, -0.5913617]).to(
                #    "cuda"
                # )
                goal_pose_torch = torch.tensor(goal_pose, device="cuda")


                cond["hand_pose"] = torch.tensor(hand_pose).to("cuda")
                cond["goal_pose"] = torch.tensor(goal_pose_torch, device="cuda")
                
                if tomm_mode:
                    cond["goal_pose"] = goal_pose_Wt   

                samples = policy(
                    cond,
                    batch_size=args.batch_size,
                    verbose=args.verbose,
                )

                sequence = utils.to_np(samples.observations[0])
                print("sequence: ", sequence)
                value = utils.to_np(samples.values_measured[0])
                last_sequence = utils.to_np(samples.observations[-1])
                end_time = time.time()
                computation_time = end_time - start_time
                fullpath = join(args.savepath, f"{t}.png")

                # Check if sequence reaches goal pose
                goal_pose_x = cond["goal_pose"][:3]

                last_q = torch.tensor(sequence[-1, 0:observation_dim]).unsqueeze(1).unsqueeze(2)
                last_pose, _ = ur5.fkine(last_q.to("cuda"))
                last_pose = last_pose[..., :3]
                d = torch.linalg.norm(goal_pose_x - last_pose)
                print(f"*******Trajectory {i} with distance: {d}")
                if plot_joint_traj:
                    # Create a plot of the actions over time
                    
                    plt.figure(1)
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
                    plt.show()
    
                    plt.figure(2)
                    plt.plot(last_sequence[:, 0], label="q0")
                    plt.plot(last_sequence[:, 1], label="q1")
                    plt.plot(last_sequence[:, 2], label="q2")
                    plt.plot(last_sequence[:, 3], label="q3")
                    plt.plot(last_sequence[:, 4], label="q4")
                    plt.plot(last_sequence[:, 5], label="q5")
                    plt.xlabel("Time step")
                    plt.ylabel("Joint angle")
                    plt.title("Diffuser output")
                    plt.legend()
                    plt.show()
                    
                save = True
                if save:
                    p_x = utils.to_np(samples.observations)
                    np.save("trajectories.npy", p_x)
                    print("trajectories saved!!")

                if d > 0.075:
                    replan = True
                    failure_count += 1
                    if plot_joint_traj:
                                           
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
                        plt.show()
                        
                    break

            cond["hand_pose"] = utils.to_np(cond["hand_pose"])
            cond["goal_pose"] = utils.to_np(cond["goal_pose"])
            next_observation, reward, terminal, _ = env.step(sequence[t, :observation_dim])
            # next_observation, reward, terminal, _ = env.step(sequence[t,0:6])
            if t == args.horizon - 1:
                time.sleep(1.0)

            else:
                time.sleep(0.010)


            if t % args.vis_freq == 0 or terminal:
                fullpath = join(args.savepath, f"{t}.png")

            if terminal:
                break

            observation = next_observation
        if not replan:
            break

    joint_position_l.append([sequence[:, 0:observation_dim]])
    hand_pose_l.append([cond["hand_pose"]])
    goal_pose_l.append([cond["goal_pose"]])
    values_l.append([value])
    computation_time_l.append([computation_time])

index = 1
directory = "logs"
while True:
    filename = join(args.savepath, f"exp{trials}_{horizon}tomm_Diffusion-Model_x_orien_middle_up.npz")
    if not exists(filename):
        np.savez(
            filename,
            joint_position=joint_position_l,
            hand_pose=hand_pose_l,
            goal_pose=goal_pose_l,
            values=values_l,
            time=computation_time_l,
        )
        print(f"Arrays saved to: {filename}")
        break
    else:
        index += 1
