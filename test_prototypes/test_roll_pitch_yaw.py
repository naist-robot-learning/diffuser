import torch
import numpy as np
''' Compute the roll, pitch, yaw angles of a x-y-z rotation sequence from an 
homogenous transformation matrix. The code was extracted and adapted to pytorch
from Peter Corke's robotics toolbox for Python.
Link: https://petercorke.github.io/robotics-toolbox-python

Args:
    H: 4x4 torch tensor

Returns:
    rpy: 3x1 torch tensor
       
'''
H = torch.randn(32,4,4).to("cuda").requires_grad_()
batch_size = H.shape[0]
rpy = torch.empty((batch_size,3), dtype=torch.float32).to('cuda')
tol = 20
eps = 1e-6

# Check for singularity
abs_r13_minus_1 = torch.abs(torch.abs(H[:, 0, 2]) - 1)
singularity_mask = abs_r13_minus_1 < tol * eps

# Compute roll (rpy[:, 0])
rpy[:, 0] = torch.where(singularity_mask, torch.tensor(0, dtype=H.dtype, device=H.device),
                        -torch.atan2(H[:, 0, 1], H[:, 0, 0]))
# Compute yaw (rpy[:, 2])
rpy[:, 2] = -torch.atan2(H[:, 1, 2], H[:, 2, 2])
mask2 = H[:, 0, 2] > 0
singularity_plus_mask2 = singularity_mask * mask2
rpy[:, 2] = torch.where(singularity_plus_mask2, torch.atan2(H[:, 2, 1], H[:, 1, 1]),
                         -torch.atan2(H[:, 1, 0], H[:, 2, 0]))
# Compute pitch (rpy[:, 1])
pitch_condition = torch.abs(H[:, 0, 2]) <= 1.0

rpy[:, 1] = torch.where(singularity_mask, torch.asin(torch.clip(H[:, 0, 2], -1.0, 1.0)),
                         -9999999)
# Handling cases where k is not 0 or 1 (k == 2 or k == 3)
k = torch.argmax(torch.abs(torch.stack((H[:, 0, 0], H[:, 0, 1], H[:, 1, 2], H[:, 2, 2]), dim=1)), dim=1)
cr = torch.cos(-torch.atan2(H[:, 0, 1], H[:, 0, 0]))
sr = torch.sin(-torch.atan2(H[:, 0, 1], H[:, 0, 0]))
sr2 = torch.sin(-torch.atan2(H[:, 1, 2], H[:, 2, 2]))
cr2 = torch.cos(-torch.atan2(H[:, 1, 2], H[:, 2, 2]))

rpy[:, 1] = torch.where(k == 0, torch.atan(H[:, 0, 2] * cr / H[:, 0, 0]), rpy[:, 1])
rpy[:, 1] = torch.where(k == 1, -torch.atan(H[:, 0, 2] * sr / H[:, 0, 1]), rpy[:, 1])
rpy[:, 1] = torch.where(k == 2, -torch.atan(H[:, 0, 2] * sr2 / H[:, 1, 2]), rpy[:, 1])
rpy[:, 1] = torch.where(k == 3, torch.atan(H[:, 0, 2] * cr2 / H[:, 2, 2]), rpy[:, 1])

rpy[0]
rpy[1]
rpy[2]