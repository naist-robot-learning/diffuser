from diffuser.robot.UR5kinematicsAndDynamics_vectorized import fkine
import sys

sys.path.insert(0, "/home/ws/src")
from CoppeliaEnv4Diffuser.gymEnvironments import CoppeliaGym, CoppeliaGymFull
import kornia
import numpy as np
import torch

torch.set_printoptions(sci_mode=False, precision=6)

# Rotation around Z
ones = torch.ones(1)
# Quaternion for negative 90 degrees rotation around z-axis on tomm base
theta = -torch.pi / 2 * ones  # 90 degrees in radians
st = torch.sin(theta / 2)
ct = torch.cos(theta / 2)
zeros = torch.zeros(1)
# Kornia is stupid, thus (w,x,y,z)
# quat_z = torch.stack([zeros, zeros, st, ct], dim=-1)
quat_zb = torch.stack([ct, zeros, zeros, st], dim=-1)
# Convert the quaternions to rotation objects
R_bprime_b = kornia.geometry.quaternion_to_rotation_matrix(quat_zb).to("cuda")


#########  TOMM World to Arm base quaternion [w,x,y,z,] #####################
quat = torch.tensor([0.321387, 0.863077, -0.321389, 0.220269], device="cuda")
R_tw_bprime = kornia.geometry.quaternion_to_rotation_matrix(quat)

# Combine the two rotations
R_tw_b = torch.matmul(R_tw_bprime, R_bprime_b).to("cuda")
# Initial position of arm base link wrt to world frame of TOMM.
p_tw = torch.tensor([0.41614, -0.225, 1.2627], dtype=torch.float32).to("cuda")
H_tw_b = torch.zeros((1, 4, 4), dtype=torch.float32, device="cuda")
H_tw_b[:, :3, :3] = R_tw_b
H_tw_b[:, :3, -1] = p_tw
H_tw_b[:, 3, -1] = 1.0


joint_angles_init = torch.tensor(
    [1.60856955, -1.68952538, -1.40157676, 2.2331241, 1.169378, 2.23968083], device="cuda"
).unsqueeze(1)


o, H_b_ee = fkine(joint_angles_init)

import ipdb

ipdb.set_trace()
