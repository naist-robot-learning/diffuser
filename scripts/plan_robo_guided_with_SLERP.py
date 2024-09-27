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
from scipy.spatial.transform import Rotation as R, Slerp

sys.path.insert(0, "/home/ws/src")
from CoppeliaEnv4Diffuser.gymEnvironments import CoppeliaGym, CoppeliaGymFull
import matplotlib.pyplot as plt
import diffuser.robot.UR5kinematicsAndDynamics_vectorized as ur5


class Parser(utils.Parser):
    dataset: str = "tomm_coppeliasim_full_path"  # Change in sequence.py also
    config: str = "config.robo"


import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline


def slerp_interpolate(pose_start, pose_goal, steps):
    """
    SLERP interpolation between two poses.
    Args:
    - pose_start: np.array, [x, y, z, qx, qy, qz, qw]
    - pose_goal: np.array, [x, y, z, qx, qy, qz, qw]
    - steps: int, number of interpolation steps

    Returns:
    - trajectory: list of interpolated poses
    """
    # Extract positions and quaternions
    pos_start = pose_start[:3]
    quat_start = pose_start[3:]
    pos_goal = pose_goal[:3]
    quat_goal = pose_goal[3:]

    # Position linear interpolation
    pos_trajectory = np.linspace(pos_start, pos_goal, steps)

    # Create a keyframe times array for interpolation [0, 1]
    key_times = [0, 1]

    # Create a Slerp object for the rotations
    slerp = Slerp(key_times, R.from_quat([quat_start, quat_goal]))

    # Interpolate rotations over the desired number of steps
    interp_times = np.linspace(0, 1, steps)
    quat_trajectory = slerp(interp_times).as_quat()

    # Combine position and quaternion trajectories
    trajectory = [np.hstack((pos, quat)) for pos, quat in zip(pos_trajectory, quat_trajectory)]
    return trajectory


def trapezoidal_velocity_profile(total_steps, max_velocity, acceleration_ratio=0.3):
    """
    Generates a trapezoidal velocity profile.
    Args:
    - total_steps: int, total number of trajectory steps
    - max_velocity: float, maximum velocity
    - acceleration_ratio: float, ratio of acceleration phase duration to total steps

    Returns:
    - velocities: list of velocities at each step
    """
    acceleration_steps = int(acceleration_ratio * total_steps)
    cruise_steps = total_steps - 2 * acceleration_steps
    velocities = np.concatenate(
        [
            np.linspace(0, max_velocity, acceleration_steps),  # Acceleration phase
            np.full(cruise_steps, max_velocity),  # Constant velocity phase
            np.linspace(max_velocity, 0, acceleration_steps),  # Deceleration phase
        ]
    )
    return velocities


def angle_difference(angle1, angle2):
    """
    Compute the shortest difference between two angles on a circle.

    Args:
    - angle1: float, first angle in radians
    - angle2: float, second angle in radians

    Returns:
    - diff: float, difference between the angles in radians, in the range [-pi, pi]
    """
    return np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2))


def compute_pose_difference_rpy(desired_pose, current_pose):
    """
    Compute the pose difference in Cartesian space, with orientation difference in RPY rates.

    Args:
    - desired_pose: np.array, desired end-effector pose [x, y, z, qx, qy, qz, qw]
    - current_pose: np.array, current end-effector pose [x, y, z, qx, qy, qz, qw]

    Returns:
    - dx: np.array, pose difference [dx, dy, dz, d_roll, d_pitch, d_yaw]
    """
    # Compute position difference (XYZ)
    pos_diff = desired_pose[:3] - current_pose[:3]

    # Convert quaternions to RPY (roll, pitch, yaw) angles
    rot_desired = R.from_quat(desired_pose[3:])
    rot_current = R.from_quat(current_pose[3:])

    rpy_desired = rot_desired.as_euler("zyx")  # RPY angles from quaternion
    rpy_current = rot_current.as_euler("zyx")  # RPY angles from quaternion

    rpy_diff = np.array(
        [
            angle_difference(rpy_desired[0], rpy_current[0]),  # Roll difference
            angle_difference(rpy_desired[1], rpy_current[1]),  # Pitch difference
            angle_difference(rpy_desired[2], rpy_current[2]),  # Yaw difference
        ]
    )

    # Combine position and orientation differences
    dx = np.hstack((pos_diff, rpy_diff))

    return dx


def compute_joint_trajectory(cartesian_trajectory, jacobian_fn, jacobian_pseudo_inv_fn, initial_joint_angles):
    """
    Computes the joint trajectory given a Cartesian trajectory using closed-loop inverse kinematics.
    Args:
    - cartesian_trajectory: list of Cartesian poses (interpolated positions and quaternions)
    - jacobian_fn: function that computes the Jacobian matrix for a given joint configuration
    - jacobian_pseudo_inv_fn: function that computes the pseudoinverse of a Jacobian
    - initial_joint_angles: np.array, initial joint configuration

    Returns:
    - joint_trajectory: list of joint angles
    """
    import ipdb

    ipdb.set_trace()
    joint_trajectory = [initial_joint_angles.unsqueeze(1).cpu().numpy()]
    current_joint_angles = initial_joint_angles.clone().unsqueeze(1)
    iterations = 0
    for pose in cartesian_trajectory:
        while True:

            # Desired end-effector pose change (dx)
            current_pose, H = ur5.fkine(current_joint_angles)  # Function to compute FK
            current_pose_npy = current_pose.cpu().numpy()

            # Convert current pose to [x, y, z, qx, qy, qz, qw]
            current_pose_quat = np.hstack(
                (current_pose_npy[0, :3], R.from_euler("zyx", current_pose_npy[0, 3:]).as_quat())
            )
            dx = compute_pose_difference_rpy(pose, current_pose_quat) * 0.5
            # Compute Jacobian and its pseudoinverse

            J = jacobian_fn(current_joint_angles)
            J_pseudo_inv = jacobian_pseudo_inv_fn(J.cpu().numpy())

            # Compute joint angles change (dq)
            dq = J_pseudo_inv @ dx[:6]  # Only use position and orientation, exclude redundancy

            # Update joint angles
            np.set_printoptions(suppress=True, precision=6)
            torch.set_printoptions(sci_mode=False, precision=6)
            current_joint_angles += torch.tensor(dq.flatten(), device="cuda").unsqueeze(1)
            # import ipdb

            # ipdb.set_trace()
            if np.linalg.norm(dx) > 0.01:
                iterations += 1
                if iterations > 10000:
                    print("CLIK unsuccesful")
            else:
                joint_trajectory.append(current_joint_angles.cpu().numpy())
                break

    return joint_trajectory


def trapezoidal_joint_trajectory(
    pose_start, pose_goal, initial_joint_angles, jacobian_fn, jacobian_pseudo_inv_fn, total_steps=39
):
    """
    Main function to compute the joint trajectory using trapezoidal velocity profile and SLERP.
    Args:
    - pose_start: np.array, [x, y, z, qx, qy, qz, qw], start pose
    - pose_goal: np.array, [x, y, z, qx, qy, qz, qw], goal pose
    - initial_joint_angles: np.array, initial joint configuration
    - jacobian_fn: function to compute the Jacobian matrix for a given joint configuration
    - jacobian_pseudo_inv_fn: function to compute the pseudoinverse of the Jacobian
    - total_steps: int, total number of steps for the trajectory

    Returns:
    - joint_trajectory: list of joint configurations
    """
    # Step 1: SLERP interpolation in Cartesian space
    cartesian_trajectory = slerp_interpolate(pose_start, pose_goal, total_steps)

    # Step 2: Generate trapezoidal velocity profile
    velocities = trapezoidal_velocity_profile(total_steps, max_velocity=1.0)

    # Step 3: Closed-loop inverse kinematics to compute joint trajectory
    joint_trajectory = compute_joint_trajectory(
        cartesian_trajectory, jacobian_fn, jacobian_pseudo_inv_fn, initial_joint_angles
    )

    return joint_trajectory


# Example usage (assuming Jacobian functions are defined)
# jacobian_fn = your_jacobian_function
# jacobian_pseudo_inv_fn = your_jacobian_pseudo_inverse_function
# initial_joint_angles = np.zeros(6)  # Assuming 6-DOF robot
# pose_start = np.array([0.5, 0.0, 0.5, 0, 0, 0, 1])
# pose_goal = np.array([1.0, 0.0, 0.5, 0, 0, 0, 1])
# joint_trajectory = trapezoidal_joint_trajectory(pose_start, pose_goal, initial_joint_angles, jacobian_fn, jacobian_pseudo_inv_fn)


# ---------------------------------- setup ----------------------------------#

args = Parser().parse_args("plan")
env = CoppeliaGymFull()

horizon = 40

# ---------------------------------- main loop ----------------------------------#
failure_count = 0
try_replan = False
joint_position_l = []
hand_pose_l = []
goal_pose_l = []
values_l = []
computation_time_l = []
trials = 1
tomm_mode = True
state_type = "path"
for i in range(0, trials):

    # #########TOMM#############
    if tomm_mode:
        ######### Starting position #############
        start_pos_Wt = torch

        # Overwrite corner point
        start_pos_Wt = torch.tensor(
            # [0.70307, -0.65419, 0.932277],
            # device="cuda",
            [0.70307, -0.65419, 1.13277],
            device="cuda",  # always -0.18 on z  # middle_up start
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
        env.original_env.set_start_pose(copp_start)
        ######### Goal Position
        goal_pos_Wt = torch.tensor(
            [
                # [1.1239, -0.651284, 0.9078644],  # right traj 2
                # device="cuda",
                # [1.1233, -0.35523, 0.9073], device="cuda"  # 65       # middle_up
                1.1239,
                -0.351284,
                0.8644,
            ],
            device="cuda",
        )
        # goal_pos_Wt = torch.tensor([1.0912, -0.56163, 1.0639], device="cuda")

        ## tilted modified -0.7210394, -0.6710702, -0.1134626, 0.1299737
        goal_quat_Wt = torch.tensor([0.6532815, 0.6532815, 0.2705981, -0.2705981], device="cuda")
        # torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
        # torch.tensor([-0.5630083, -0.5206123, -0.4305665, 0.476022], device="cuda")
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
        env.original_env.set_goal_pose(copp_goal)

    # env.original_env.set_goal_pose(np.array([0.55, 0.0, 0.1, 0.7071, 0.7071, 0.0, 0.0]))  # C14
    ############ UR5 Normal table configuration #################
    # env.original_env.set_start_pose(np.array([0.15, 0.2, 0.3, 0.7071, 0.7071, 0.0, 0.0]))  # C10
    # env.original_env.set_goal_pose(np.array([0.55, 0.0, 0.1, 0.7071, 0.7071, 0.0, 0.0]))  # C14

    # env.original_env.set_random_goal_pose(xy_only=True, tomm_mode=tomm_mode)
    observation, goal_pose, hand_pose = env.reset(
        state_type=state_type
    )  ##### goal_pose is with respect to the UR5 CoppeliaSim Base
    print("*******observation********\n", observation)
    # goal_pose = np.array([0.55, 0.2, 0.1, 0.707, 0.707, 0, 0])
    ## observations for rendering
    rollout = [observation.copy()]
    cond = {}
    total_reward = 0
    if i % 10 == 0 and i != 0:
        print(f"Failed to reach goal pose {failure_count} times so far")

    while True:
        replan = False
        for t in range(0, 40):
            # print("t: ", t)
            state = observation.copy()
            print("t: ", t)
            ## can replan if desired, but the open-loop plans are good enough for maze2d
            ## that we really only need to plan once
            if t == 0:
                computation_time = 0
                start_time = time.time()
                print(f"Planning experiment #{i}")
                initial_joint_angles = torch.tensor(observation).to("cuda")
                # cond[0] = torch.tensor([0.9990606, -0.91376438, -2.03536981, -1.68044293, 1.9546568, -0.5913617]).to(
                #    "cuda"
                # )
                # goal_pose_torch = torch.tensor(goal_pose, device="cuda")

                # Hard goal set with respect to RVIZ setup

                # hard_goal_pose_torch = torch.tensor([0.96107, -0.59474, 0.90507, 0.0, 0.0, 0.0, 1.0], device="cuda")
                # goal_quat =
                # Define the rotation from the new frame to the world frame (-90 degrees about z-axis)
                # rotation_world_to_new = R.from_euler('z', -90, degrees=True)

                # Convert the goal quaternion to a scipy Rotation object
                # goal_rotation_world = R.from_quat(goal_quaternion_world)

                # Combine the rotations: first apply the rotation from the new frame to the world frame,
                # then apply the goal rotation (which is in the world frame)
                # goal_rotation_new_frame = rotation_world_to_new * goal_rotation_world

                # Get the quaternion with respect to the new frame
                # goal_quaternion_new_frame = goal_rotation_new_frame.as_quat()

                cond["hand_pose"] = torch.tensor(hand_pose).to("cuda")
                # cond["goal_pose"] = torch.tensor(goal_pose_torch, device="cuda")
                cond["goal_pose"] = goal_pose_Wt
                # cond[47] = torch.tensor([0.16024063, -1.76447969, -1.85052044, -1.07379095, 1.62734438, -1.41637277])

                # cond[16] = torch.tensor([0.15, 0.40, 0.38, 0.0, 0.0, 0.0, 0.0]).to("cuda")
                # cond[32] = torch.tensor([0.30, 0.10, 0.38, 0.0, 0.0, 0.0, 0.0]).to("cuda")
                # print("observation: ", observation)
                # import ipdb; ipdb.set_trace()
                # action, samples = policy(cond, batch_size=args.batch_size)
                # action, samples = policy(cond, batch_size=args.batch_size, verbose=args.verbose)

                pose_start = torch.concatenate([start_pos_Wt, start_quat_Wt], axis=-1).cpu().numpy()
                pose_goal = torch.concatenate([goal_pos_Wt, goal_quat_Wt], axis=-1).cpu().numpy()

                samples = trapezoidal_joint_trajectory(
                    pose_start, pose_goal, initial_joint_angles, ur5.compute_analytical_jacobian, np.linalg.pinv
                )

                sequence = np.array(samples).squeeze()
                import ipdb

                ipdb.set_trace()
                # print("sequence: ", sequence)

                last_sequence = samples[-1]
                end_time = time.time()
                computation_time = end_time - start_time
                # fullpath = join(args.savepath, f"{t}.png")
                # Create a plot of the actions over time
                # plt.figure(1)
                # plt.plot(sequence[:, 0], label="q0")
                # plt.plot(sequence[:, 1], label="q1")
                # plt.plot(sequence[:, 2], label="q2")
                # plt.plot(sequence[:, 3], label="q3")
                # plt.plot(sequence[:, 4], label="q4")
                # plt.plot(sequence[:, 5], label="q5")
                # plt.show()
                # Check if sequence reaches goal pose
                goal_pose_x = cond["goal_pose"][:3]
                observation_dim = len(state)
                last_q = torch.tensor(sequence[-1]).unsqueeze(1)
                last_pose, _ = ur5.fkine(last_q.to("cuda"))
                last_pose = last_pose[..., :3]
                d = torch.linalg.norm(goal_pose_x - last_pose)
                print(f"*******Trajectory {i} with distance: {d}")

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
                save = True
                if save:
                    pass
                    # p_x = utils.to_np(samples)
                    # np.save("trajectories.npy", p_x)
                    # print("trajectories saved!!")

                if d > 0.1:
                    replan = True
                    failure_count += 1
                    # Create a plot of the actions over time
                    # plt.figure(i)
                    # plt.plot(sequence[:, 0], label="q0")
                    # plt.plot(sequence[:, 1], label="q1")
                    # plt.plot(sequence[:, 2], label="q2")
                    # plt.plot(sequence[:, 3], label="q3")
                    # plt.plot(sequence[:, 4], label="q4")
                    # plt.plot(sequence[:, 5], label="q5")

                    # plt.xlabel("Time step")
                    # plt.ylabel("Joint angle")
                    # plt.title("Diffuser output")
                    # plt.legend()
                    # plt.savefig(f"failure{i}.png")
                    print("**************REPLANNING*****************")
                    # plt.show()
                    break

            # renderer.composite(fullpath, samples.observations, ncol=1)
            # import ipdb; ipdb.set_trace()
            # next_observation, reward, terminal, _ = env.step(actions[t, :])

            next_observation, reward, terminal, _ = env.step(sequence[t])
            # next_observation, reward, terminal, _ = env.step(sequence[t,0:6])
            if t == horizon - 1:
                time.sleep(1.0)

            else:
                time.sleep(0.010)

            # 0.004 250Hz
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
        if not replan:
            break

    joint_position_l.append(sequence)
    # goal_pose_l.append([sequence[:, 6:13]])

    computation_time_l.append([computation_time])

# logger.finish(t, env.max_episode_steps, score=score, value=0)
index = 1
directory = "logs"
while True:
    filename = join(args.savepath, f"exp{trials}_{horizon}tomm_LinearInterpolation_middle.npz")
    if not exists(filename):
        np.savez(
            filename,
            joint_position=np.array(joint_position_l).squeeze(),
            hand_pose=hand_pose_l,
            goal_pose=goal_pose_l,
            values=values_l,
            time=computation_time_l,
        )
        print(f"Arrays saved to: {filename}")
        break
    else:
        index += 1
