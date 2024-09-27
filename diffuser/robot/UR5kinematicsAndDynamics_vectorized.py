import einops
import kornia
import numpy as np
import time
import torch
import math
from diffuser.robot.urdf_mass_matrix_UR5 import compute_mass_matrix_urdf

torch.pi = math.pi


def dh_to_transformation(a, alpha, d, theta):
    """Convert DH parameters to a homogenous transformation matrix

    Args:
        a (float): link length
        alpha (float): link twist
        d (float): link offset
        theta (float): joint angle

    Returns:
        T (4x4 torch tensor): homogenous transformation matrix

    """
    horizon = theta.shape[0]
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    cos_alpha, sin_alpha = torch.cos(alpha), torch.sin(alpha)
    T = torch.eye(4, dtype=torch.float32, device="cuda").unsqueeze(0).expand(horizon, -1, -1).clone()

    T[:, 0, 0] = cos_theta
    T[:, 0, 1] = -sin_theta * cos_alpha
    T[:, 0, 2] = sin_theta * sin_alpha
    T[:, 0, 3] = a * cos_theta
    T[:, 1, 0] = sin_theta
    T[:, 1, 1] = cos_theta * cos_alpha
    T[:, 1, 2] = -cos_theta * sin_alpha
    T[:, 1, 3] = a * sin_theta
    T[:, 2, 1] = sin_alpha
    T[:, 2, 2] = cos_alpha
    T[:, 2, 3] = d
    return T


def compute_fk(q):
    """Computes the forward kinematics of the UR5 manipulator

    Args:
        q (6x1 torch tensor): joint angles

    Returns:
        T (4x4 torch tensor): homogenous transformation matrix

    """
    tomm_mode = False
    # UR5 Standard DH parameters (a, alpha, d, theta)
    DH_params = torch.tensor(
        [
            [0.0, torch.pi / 2, 0.08946],  # link 1 parameters
            [-0.425, 0.0, 0.0],  # link 2 parameters
            [-0.39225, 0.0, 0.0],  # link 3 parameters
            [0.0, torch.pi / 2, 0.10915],  # link 4 parameters
            [0.0, -torch.pi / 2, 0.09465],  # link 5 parameters
            [0.0, 0.0, 0.0823 + 0.13385],  # link 6 parameters + robotiQ hand offset
        ]
    ).to("cuda")
    horizon = q.shape[1]
    assert len(q) == 6, "There should be 6 joint angles"
    T = torch.zeros((horizon, 4, 4), dtype=torch.float32, device="cuda")
    ones = torch.ones(horizon)
    ################################## Tomm required offsets [x,y,z,w]####################################
    if tomm_mode:
        # print("tomm_mode on")

        H_Wt_bC, _ = get_tomm_setup(horizon)
        T[:, :3, :3] = H_Wt_bC[:, :3, :3]
        T[:, :3, -1] = H_Wt_bC[:, :3, -1]
        T[:, 3, -1] = ones
    else:
        T[:, :3, :3] = torch.eye(3, dtype=torch.float32, device="cuda")
        T[:, 3, -1] = ones
    for i, dh in enumerate(DH_params):
        a = dh[0]
        alpha = dh[1]
        d = dh[2]
        if i == 5 and tomm_mode:  ## Stupid Gripper shift
            shift = q[i, :] - torch.pi / 2
            T = torch.matmul(T, dh_to_transformation(a, alpha, d, shift))
        else:
            T = torch.matmul(T, dh_to_transformation(a, alpha, d, q[i, :]))

    return T


def compute_rpy_xyz(H):
    """Compute the roll, pitch, yaw angles of a x-y-z rotation sequence from an
    homogenous transformation matrix. The code was extracted and adapted to pytorch
    from Peter Corke's robotics toolbox for Python.
    Link: https://petercorke.github.io/robotics-toolbox-python

    Args:
        H: 4x4 torch tensor

    Returns:
        rpy: 3x1 torch tensor

    """
    horizon = H.shape[0]
    rpy = torch.zeros((horizon, 3), dtype=torch.float32).to("cuda")
    tol = 20
    eps = 1e-6

    # if abs(abs(H[0, 2]) - 1) < tol * eps:  # when |R13| == 1
    #     # singularity
    #     rpy[0] = 0  # roll is zero
    #     if H[0, 2] > 0:
    #         rpy[2] = torch.atan2(H[2, 1], H[1, 1])  # R+Y
    #     else:
    #         rpy[2] = -torch.atan2(H[1, 0], H[2, 0])  # R-Y
    #     rpy[1] = torch.asin(torch.clip(H[0, 2], -1.0, 1.0))
    # else:
    #     rpy[0] = -torch.atan2(H[0, 1], H[0, 0])
    #     rpy[2] = -torch.atan2(H[1, 2], H[2, 2])
    #     k = np.argmax(torch.abs(torch.tensor([H[0, 0], H[0, 1], H[1, 2], H[2, 2]])))
    #     rpy = rpy.clone()
    #     if k == 0:
    #         cr = torch.cos(-torch.atan2(H[0, 1], H[0, 0]))
    #         rpy[1] = torch.atan(H[0, 2] * cr / H[0, 0])
    #     elif k == 1:
    #         sr = torch.sin(-torch.atan2(H[0, 1], H[0, 0]))
    #         rpy[1] = -torch.atan(H[0, 2] * sr / H[0, 1])
    #     elif k == 2:
    #         sr = torch.sin(-torch.atan2(H[1, 2], H[2, 2]))
    #         rpy[1] = -torch.atan(H[0, 2] * sr / H[1, 2])
    #     elif k == 3:
    #         cr = torch.cos(-torch.atan2(H[1, 2], H[2, 2]))
    #         rpy[1] = torch.atan(H[0, 2] * cr / H[2, 2])
    horizon = H.shape[0]
    rpy = torch.zeros((horizon, 3), dtype=torch.float32).to("cuda")
    tol = 20
    eps = 1e-6

    # Check for singularity
    abs_r13_minus_1 = torch.abs(torch.abs(H[:, 0, 2]) - 1)
    singularity_mask = abs_r13_minus_1 < tol * eps

    # Compute roll (rpy[:, 0])
    rpy[:, 0] = torch.where(
        singularity_mask,
        torch.tensor(0, dtype=H.dtype, device=H.device),
        -torch.atan2(H[:, 0, 1], H[:, 0, 0]),
    )
    # Compute yaw (rpy[:, 2])
    # rpy[:, 2] = -torch.atan2(H[:, 1, 2], H[:, 2, 2])
    mask2 = H[:, 0, 2] > 0
    rpy[:, 2] = torch.where(
        singularity_mask,
        torch.where(mask2, torch.atan2(H[:, 2, 1], H[:, 1, 1]), -torch.atan2(H[:, 1, 0], H[:, 2, 0])),
        -torch.atan2(H[:, 1, 2], H[:, 2, 2]),
    )

    # Compute pitch (rpy[:, 1])
    rpy[:, 1] = torch.where(singularity_mask, torch.asin(torch.clip(H[:, 0, 2], -1.0, 1.0)), -9999999)

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

    return rpy[:, 0], rpy[:, 1], rpy[:, 2]


def hom_to_pose(H):
    """Converts a 4x4 homogenous transformation matrix to a 6x1 compact pose

    Args:
        H (4x4 torch tensor): homogenous transformation matrix

    Returns:
        pose (6x1 torch tensor): [x, y, z, roll, pitch, yaw]

    """
    horizon = H.shape[0]
    pose = torch.zeros((horizon, 6), dtype=torch.float32).to("cuda")
    pose[:, 0] = H[:, 0, 3]
    pose[:, 1] = H[:, 1, 3]
    pose[:, 2] = H[:, 2, 3]
    pose[:, 3], pose[:, 4], pose[:, 5] = compute_rpy_xyz(H)

    return pose


def fkine(q):
    """Computes the forward kinematics of the UR5 manipulator
    in compact representation [x, y, z, roll, pitch, yaw]

    Args:
        q (6x(B H) torch tensor): joint angles

    Returns:
        compact (6x(B H) torch tensor): [x, y, z, roll, pitch, yaw]

    """
    H = compute_fk(q)
    compact = hom_to_pose(H)
    return compact, H


def compute_analytical_jacobian(q):
    horizon = q.shape[1]
    timer = False

    def __fkine__(inputs):
        fk, _ = fkine(inputs)
        return fk.sum(axis=0)

    if timer:
        start_time = time.time()
        outputs = fkine(q)
        end_time = time.time()
        print("Forward kinematics time: ", end_time - start_time)
        start_time = time.time()
        jac = torch.autograd.functional.jacobian(
            __fkine__, q, retain_graph=True, create_graph=False, vectorize=True
        ).to("cuda")
        end_time = time.time()
        print("Jacobian computation time: ", end_time - start_time)
        start_time = time.time()
        J = einops.rearrange(jac, "m n k -> k m n")
        end_time = time.time()
        print("Jacobian rearrange time: ", end_time - start_time)
    else:
        jac = torch.autograd.functional.jacobian(__fkine__, q, create_graph=False, vectorize=True).to("cuda")
        J = einops.rearrange(jac, "m n k -> k m n").to("cuda")

    return J


def compute_inertia_matrix(q):
    """Computes the inertia matrix of the UR5 manipulator based on
    https://github.com/kkufieta/ur5_modeling_force_estimate and
    adapted to pytorch.

    Matrix structure

    Mq = [m11 m12 m13 m14 m15 m16;
         m21 m22 m23 m24 m25 m26;
         m31 m32 m33 m34 m35 m36;
         m41 m42 m43 m44 m45 m46;
         m51 m52 m53 m54 m55 m56;
         m61 m62 m63 m64 m65 m66];

    Args:
        q (6xBatch torch tensor): joint angles

    Returns:
        Mq (Batchx6x6 torch tensor): inertia matrix
    """

    return compute_mass_matrix_urdf(q)
    ## Returns the dynamics of the UR5 manipulator
    horizon = q.shape[1]
    Mq = torch.empty(horizon, 6, 6, dtype=torch.float32).to("cuda")
    q1 = q[0, :]
    q2 = q[1, :]
    q3 = q[2, :]
    q4 = q[3, :]
    q5 = q[4, :]
    q6 = q[5, :]

    cos_q234_2q5 = torch.cos(q2 + q3 + q4 + 2.0 * q5)
    cos_2q5_neg342 = torch.cos(2.0 * q5 - q3 - q4 - q2)
    cos_q5neg3neg4neg2 = torch.cos(q5 - q3 - q4 - q2)
    cos_q2345 = torch.cos(q2 + q3 + q4 + q5)
    cos_q234 = torch.cos(q2 + q3 + q4)
    sin_q234 = torch.sin(q2 + q3 + q4)
    sin_q235 = torch.sin(q2 + q3 + q5)
    sin_q23neg5 = torch.sin(q2 + q3 - q5)
    sin_q45 = torch.sin(q4 + q5)
    sin_q25 = torch.sin(q2 + q5)
    sin_q2_neg5 = torch.sin(q2 - q5)
    sin_q34 = torch.sin(q3 + q4)
    sin_q23 = torch.sin(q2 + q3)
    cos_q5 = torch.cos(q5)
    sin_q5 = torch.sin(q5)
    cos_q4 = torch.cos(q4)
    sin_q4 = torch.sin(q4)
    cos_q3 = torch.cos(q3)
    sin_q3 = torch.sin(q3)
    sin_q2 = torch.sin(q2)

    # Mq[:,0,0] = m11 = 0.007460552829*torch.sin(q3 + q4 + q5) - 0.0002254000336*torch.cos(2.0*q2 + 2.0*q3 + 2.0*q4 + 2.0*q5) + 0.007460552829*torch.sin(2.0*q2 + q3 + q4 + q5) + 0.7014282607*torch.cos(2.0*q2 + q3) - 0.007460552829*torch.sin(q3 + q4 - q5) - 0.007460552829*torch.sin(2.0*q2 + q3 + q4 - q5) - 0.05830173968*torch.sin(2.0*q2 + 2.0*q3 + q4) + 0.001614617887*torch.cos(2.0*q2 + 2.0*q3 + 2.0*q4 + q5) + 0.8028639871*torch.cos(2.0*q2) + 0.0004508000672*torch.cos(2.0*q5) - 0.001614617887*torch.cos(2.0*q2 + 2.0*q3 + 2.0*q4 - q5) - 0.0002254000336*torch.cos(2.0*q2 + 2.0*q3 + 2.0*q4 - 2.0*q5) - 0.063140533*sin_q34 + 0.006888811168*sin_q45 - 0.063140533*torch.sin(2.0*q2 + q3 + q4) - 0.006888811168*torch.sin(q4 - q5) + 0.7014282607*cos_q3 + 0.00765364949*cos_q5 + 0.006888811168*torch.sin(2.0*q2 + 2.0*q3 + q4 + q5) + 0.3129702942*torch.cos(2.0*q2 + 2.0*q3) - 0.05830173968*sin_q4 - 0.005743907935*torch.cos(2.0*q2 + 2.0*q3 + 2.0*q4) - 0.006888811168*torch.sin(2.0*q2 + 2.0*q3 + q4 - q5) + 1.28924888
    # Mq[:,0,1] = m12 = 0.01615783641*cos_q234 + 0.006888811168*sin_q235 - 0.0004508000672*cos_q234_2q5 + 0.006888811168*sin_q23neg5 + 0.00352803026*cos_q5neg3neg4neg2 + 0.0004508000672*cos_2q5_neg342 - 0.0002987944852*cos_q2345 + 0.1313181732*sin_q23 + 0.007460552829*sin_q25 + 0.007460552829*sin_q2_neg5+ 0.348513447*sin_q2
    # Mq[:,0,2] = m13 = 0.01615783641*cos_q234 + 0.006888811168*sin_q235 - 0.0004508000672*cos_q234_2q5 + 0.006888811168*sin_q23neg5 + 0.00352803026*cos_q5neg3neg4neg2 + 0.0004508000672*cos_2q5_neg342 - 0.0002987944852*cos_q2345 + 0.1313181732*sin_q23
    # Mq[:,0,3] = m14 = 0.01615783641*cos_q234 - 0.0004508000672*cos_q234_2q5 + 0.00352803026*cos_q5neg3neg4neg2 + 0.0004508000672*cos_2q5_neg342 - 0.0002987944852*cos_q2345
    # Mq[:,0,4] = m15 = 0.006888811168*sin_q23neg5 - 0.006888811168*sin_q235 - 0.002840479501*cos_q234 - 0.0002987944852*cos_q5neg3neg4neg2 - 0.00352803026*cos_q2345 - 0.007460552829*sin_q25 + 0.007460552829*sin_q2_neg5
    # Mq[:,0,5] = m16 = -0.000138534912*sin_q234*sin_q5
    # Mq[:,1,0] = m21 = 0.01615783641*cos_q234 + 0.006888811168*sin_q235 - 0.0004508000672*cos_q234_2q5 + 0.006888811168*sin_q23neg5 + 0.00352803026*torch.cos(q5 - q3 - q4 - q2) + 0.0004508000672*cos_2q5_neg342 - 0.0002987944852*cos_q2345 + 0.1313181732*sin_q23 + 0.007460552829*sin_q25 + 0.007460552829*sin_q2_neg5+ 0.348513447*sin_q2
    # Mq[:,1,1] = m22 = 1.402856521*cos_q3 - 0.1166034794*sin_q4 - 0.126281066*cos_q3*sin_q4 - 0.126281066*cos_q4*sin_q3 + 0.02755524467*cos_q4*sin_q5 - 0.001803200269*cos_q5**2 - 0.02984221131*sin_q3*sin_q4*sin_q5 + 0.02984221131*cos_q3*cos_q4*sin_q5 + 2.263576776
    # Mq[:,1,2] = m23 = 0.7014282607*cos_q3 - 0.1166034794*sin_q4 - 0.063140533*cos_q3*sin_q4 - 0.063140533*cos_q4*sin_q3 + 0.02755524467*cos_q4*sin_q5 - 0.001803200269*cos_q5**2 - 0.01492110566*sin_q3*sin_q4*sin_q5 + 0.01492110566*cos_q3*cos_q4*sin_q5 + 0.644094696
    # Mq[:,1,3] = m24 = 0.01377762234*cos_q4*sin_q5 - 0.063140533*cos_q3*sin_q4 - 0.063140533*cos_q4*sin_q3 - 0.05830173968*sin_q4 - 0.001803200269*cos_q5**2 - 0.01492110566*sin_q3*sin_q4*sin_q5 + 0.01492110566*cos_q3*cos_q4*sin_q5 + 0.01612863983
    # Mq[:,1,4] = m25 = cos_q5*(0.01492110566*sin_q34 + 0.01377762234*sin_q4 - 0.003229235775)
    # Mq[:,1,5] = m26 = 0.000138534912*cos_q5
    # Mq[:,2,0] = m31 = 0.01615783641*cos_q234 + 0.006888811168*sin_q235 - 0.0004508000672*cos_q234_2q5 + 0.006888811168*sin_q23neg5 + 0.00352803026*cos_q5neg3neg4neg2 + 0.0004508000672*cos_2q5_neg342 - 0.0002987944852*cos_q2345 + 0.1313181732*sin_q23
    # Mq[:,2,1] = m32 = 0.7014282607*cos_q3 - 0.1166034794*sin_q4 - 0.063140533*cos_q3*sin_q4 - 0.063140533*cos_q4*sin_q3 + 0.02755524467*cos_q4*sin_q5 - 0.001803200269*cos_q5**2 - 0.01492110566*sin_q3*sin_q4*sin_q5 + 0.01492110566*cos_q3*cos_q4*sin_q5 + 0.644094696
    # Mq[:,2,2] = m33 = 0.02755524467*cos_q4*sin_q5 - 0.1166034794*sin_q4 - 0.001803200269*cos_q5**2 + 0.644094696
    # Mq[:,2,3] = m34 = 0.01377762234*cos_q4*sin_q5 - 0.05830173968*sin_q4 - 0.001803200269*cos_q5**2 + 0.01612863983
    # Mq[:,2,4] = m35 = cos_q5*(0.01377762234*sin_q4 - 0.003229235775)
    # Mq[:,2,5] = m36 = 0.000138534912*cos_q5
    # Mq[:,3,0] = m41 = 0.01615783641*cos_q234 - 0.0004508000672*cos_q234_2q5 + 0.00352803026*cos_q5neg3neg4neg2 + 0.0004508000672*cos_2q5_neg342 - 0.0002987944852*cos_q2345
    # Mq[:,3,1] = m42 = 0.01377762234*cos_q4*sin_q5 - 0.063140533*cos_q3*sin_q4 - 0.063140533*cos_q4*sin_q3 - 0.05830173968*sin_q4 - 0.001803200269*cos_q5**2 - 0.01492110566*sin_q3*sin_q4*sin_q5 + 0.01492110566*cos_q3*cos_q4*sin_q5 + 0.01612863983
    # Mq[:,3,2] = m43 = 0.01377762234*cos_q4*sin_q5 - 0.05830173968*sin_q4 - 0.001803200269*cos_q5**2 + 0.01612863983
    # Mq[:,3,3] = m44 = 0.001803200269*sin_q5**2 + 0.01432543956
    # Mq[:,3,4] = m45 = -0.003229235775*cos_q5
    # Mq[:,3,5] = m46 = 0.000138534912*cos_q5
    # Mq[:,4,0] = m51 = 0.006888811168*sin_q23neg5 - 0.006888811168*sin_q235 - 0.002840479501*cos_q234 - 0.0002987944852*cos_q5neg3neg4neg2 - 0.00352803026*cos_q2345 - 0.007460552829*sin_q25 + 0.007460552829*sin_q2_neg5
    # Mq[:,4,1] = m52 = cos_q5*(0.01492110566*sin_q34 + 0.01377762234*sin_q4 - 0.003229235775)
    # Mq[:,4,2] = m53 = cos_q5*(0.01377762234*sin_q4 - 0.003229235775)
    # Mq[:,4,3] = m54 = -0.003229235775*cos_q5
    # Mq[:,4,4] = m55 = 0.002840479501
    # Mq[:,4,5] = m56 = 0
    # Mq[:,5,0] = m61 = -0.000138534912*sin_q234*sin_q5
    # Mq[:,5,1] = m62 = 0.000138534912*cos_q5
    # Mq[:,5,2] = m63 = 0.000138534912*cos_q5
    # Mq[:,5,3] = m64 = 0.000138534912*cos_q5
    # Mq[:,5,4] = m65 = 0
    # Mq[:,5,5] = m66 = 0.000138534912
    Mq[:, 0, 0] = m11 = (
        0.007460552829 * torch.sin(q3 + q4 + q5)
        - 0.0002254000336 * torch.cos(2.0 * q2 + 2.0 * q3 + 2.0 * q4 + 2.0 * q5)
        + 0.007460552829 * torch.sin(2.0 * q2 + q3 + q4 + q5)
        + 0.7014282607 * torch.cos(2.0 * q2 + q3)
        - 0.007460552829 * torch.sin(q3 + q4 - q5)
        - 0.007460552829 * torch.sin(2.0 * q2 + q3 + q4 - q5)
        - 0.05830173968 * torch.sin(2.0 * q2 + 2.0 * q3 + q4)
        + 0.001614617887 * torch.cos(2.0 * q2 + 2.0 * q3 + 2.0 * q4 + q5)
        + 0.8028639871 * torch.cos(2.0 * q2)
        + 0.0004508000672 * torch.cos(2.0 * q5)
        - 0.001614617887 * torch.cos(2.0 * q2 + 2.0 * q3 + 2.0 * q4 - q5)
        - 0.0002254000336 * torch.cos(2.0 * q2 + 2.0 * q3 + 2.0 * q4 - 2.0 * q5)
        - 0.063140533 * sin_q34
        + 0.006888811168 * sin_q45
        - 0.063140533 * torch.sin(2.0 * q2 + q3 + q4)
        - 0.006888811168 * torch.sin(q4 - q5)
        + 0.7014282607 * cos_q3
        + 0.00765364949 * cos_q5
        + 0.006888811168 * torch.sin(2.0 * q2 + 2.0 * q3 + q4 + q5)
        + 0.3129702942 * torch.cos(2.0 * q2 + 2.0 * q3)
        - 0.05830173968 * sin_q4
        - 0.005743907935 * torch.cos(2.0 * q2 + 2.0 * q3 + 2.0 * q4)
        - 0.006888811168 * torch.sin(2.0 * q2 + 2.0 * q3 + q4 - q5)
        + 1.28924888
    )
    Mq[:, 1, 1] = m22 = (
        1.402856521 * cos_q3
        - 0.1166034794 * sin_q4
        - 0.126281066 * cos_q3 * sin_q4
        - 0.126281066 * cos_q4 * sin_q3
        + 0.02755524467 * cos_q4 * sin_q5
        - 0.001803200269 * cos_q5**2
        - 0.02984221131 * sin_q3 * sin_q4 * sin_q5
        + 0.02984221131 * cos_q3 * cos_q4 * sin_q5
        + 2.263576776
    )
    Mq[:, 2, 2] = m33 = (
        0.02755524467 * cos_q4 * sin_q5 - 0.1166034794 * sin_q4 - 0.001803200269 * cos_q5**2 + 0.644094696
    )
    Mq[:, 3, 3] = m44 = 0.001803200269 * sin_q5**2 + 0.01432543956
    Mq[:, 4, 4] = m55 = 0.002840479501
    Mq[:, 5, 5] = m66 = 0.000138534912

    Mq[:, 0, 2] = Mq[:, 2, 0] = (
        0.01615783641 * cos_q234
        + 0.006888811168 * sin_q235
        - 0.0004508000672 * cos_q234_2q5
        + 0.006888811168 * sin_q23neg5
        + 0.00352803026 * cos_q5neg3neg4neg2
        + 0.0004508000672 * cos_2q5_neg342
        - 0.0002987944852 * cos_q2345
        + 0.1313181732 * sin_q23
    )  # m13 m31
    Mq[:, 0, 1] = Mq[:, 1, 0] = (
        Mq[:, 0, 2] + 0.007460552829 * sin_q25 + 0.007460552829 * sin_q2_neg5 + 0.348513447 * sin_q2
    )  # m12 m21
    Mq[:, 0, 3] = Mq[:, 3, 0] = (
        0.01615783641 * cos_q234
        - 0.0004508000672 * cos_q234_2q5
        + 0.00352803026 * cos_q5neg3neg4neg2
        + 0.0004508000672 * cos_2q5_neg342
        - 0.0002987944852 * cos_q2345
    )  # m14 m41
    Mq[:, 0, 4] = Mq[:, 4, 0] = (
        0.006888811168 * sin_q23neg5
        - 0.006888811168 * sin_q235
        - 0.002840479501 * cos_q234
        - 0.0002987944852 * cos_q5neg3neg4neg2
        - 0.00352803026 * cos_q2345
        - 0.007460552829 * sin_q25
        + 0.007460552829 * sin_q2_neg5
    )  # m15 m51
    Mq[:, 0, 5] = Mq[:, 5, 0] = -0.000138534912 * sin_q234 * sin_q5  # m16 m61
    Mq[:, 1, 2] = Mq[:, 2, 1] = (
        0.7014282607 * cos_q3
        - 0.1166034794 * sin_q4
        - 0.063140533 * cos_q3 * sin_q4
        - 0.063140533 * cos_q4 * sin_q3
        + 0.02755524467 * cos_q4 * sin_q5
        - 0.001803200269 * cos_q5**2
        - 0.01492110566 * sin_q3 * sin_q4 * sin_q5
        + 0.01492110566 * cos_q3 * cos_q4 * sin_q5
        + 0.644094696
    )  # m23 m32
    Mq[:, 1, 3] = Mq[:, 3, 1] = (
        0.01377762234 * cos_q4 * sin_q5
        - 0.063140533 * cos_q3 * sin_q4
        - 0.063140533 * cos_q4 * sin_q3
        - 0.05830173968 * sin_q4
        - 0.001803200269 * cos_q5**2
        - 0.01492110566 * sin_q3 * sin_q4 * sin_q5
        + 0.01492110566 * cos_q3 * cos_q4 * sin_q5
        + 0.01612863983
    )  # m24 m42

    Mq[:, 1, 4] = Mq[:, 4, 1] = cos_q5 * (0.01492110566 * sin_q34 + 0.01377762234 * sin_q4 - 0.003229235775)  # m25 m52
    Mq[:, 1, 5] = Mq[:, 5, 1] = 0.000138534912 * cos_q5  # m26 m62
    Mq[:, 2, 3] = Mq[:, 3, 2] = (
        0.01377762234 * cos_q4 * sin_q5 - 0.05830173968 * sin_q4 - 0.001803200269 * cos_q5**2 + 0.01612863983
    )  # m34 m43
    Mq[:, 2, 4] = Mq[:, 4, 2] = cos_q5 * (0.01377762234 * sin_q4 - 0.003229235775)  # m35 m53
    Mq[:, 2, 5] = Mq[:, 5, 2] = 0.000138534912 * cos_q5  # m36 m63
    Mq[:, 3, 4] = Mq[:, 4, 3] = -0.003229235775 * cos_q5  # m45 m54
    Mq[:, 3, 5] = Mq[:, 5, 3] = 0.000138534912 * cos_q5  # m46 m64
    Mq[:, 4, 5] = Mq[:, 5, 4] = 0  # m56 m65

    return Mq


def compute_reflected_mass(q, u):
    """Computes the cumulative reflected mass of a trajectory from the UR5 manipulator
    along some direction u

    Args:
        q (Bx6xH) torch tensor): joint angles
        u ((B H)x3) torch tensor): direction vector

    Returns:
        mu (float): reflected mass along u
    """

    b, t, h = q.shape
    q = einops.rearrange(q, "b t h -> t (b h)").to("cuda")
    J = compute_analytical_jacobian(q)
    Mq = compute_inertia_matrix(q)
    J_T = J.permute(0, 2, 1)
    M_x_inv = (J @ torch.linalg.solve(Mq, J_T))[:, :3, :3]
    u_T = u.permute(0, 2, 1)
    mu = 1 / (torch.matmul(u_T, torch.matmul(M_x_inv, u)).squeeze())
    mu = einops.rearrange(mu, "(b h) -> b h", b=b, h=h)

    return mu


def compute_kinetic_energy_matrix(q):
    """Computes the reflected mass of the UR5 manipulator
    along some direction u

    Args:
        q (Bx6xH) torch tensor): joint angles
        u (6x(B H) torch tensor): direction vector

    Returns:
        mu (float): reflected mass along u
    """
    b, t, h = q.shape
    q = einops.rearrange(q, "b t h -> t (b h)").to("cuda")
    J = compute_analytical_jacobian(q)
    Mq = compute_inertia_matrix(q)
    J_T = J.permute(0, 2, 1)
    M_x_inv = (J @ torch.linalg.solve(Mq, J_T))[:, :3, :3]
    return M_x_inv


def get_tomm_setup(horizon):
    """Returns the transformation matrices to change from tomm world coordinates to
    Coppelia Sim base_link coordinates

    Args:
        horizon (integer): length of the Batch of resulting Transformation matrices

    Returns:
        H_Wt_bC, H_bC_Wt: Transformation matrices H_Wt_bC, H_bC_Wt
    """

    # World to Arm base quaternion [w,x,y,z,]
    quat = torch.tensor([0.321387, 0.863077, -0.321389, 0.220269], device="cuda")
    ############# TOMM #####################
    #  quat represents the quaternion from TOMM world to the base in Rviz,
    #  The coppeliaSim UR5 base_link is shifted counterclockwise 90 degrees about z with respect
    #  to the RVIZ model. This had to be done since the RVIZ URDF has a 90 degree shift on coordinate
    #  frame of the base_link. So
    #  H_Wt_bR @ H_bR_bC @ H_bC_eeC   -> For Reasoning about pose of visualization in terms of the World fram
    #
    #  H_bC_Wt @ H_Wt_eeR   -> For inputing poses to Coppelia Env based on the coordinates of RVIZ
    #  R -> Rviz
    #  C -> CoppeliaSim
    #  b -> Base_link
    #  ee-> end-effector
    ###########################################################################################
    R_Wt_bR = kornia.geometry.quaternion_to_rotation_matrix(quat)

    # Quaternion for 90 degrees rotation around z-axis
    ones = torch.ones(horizon)
    theta = -torch.pi / 2 * ones  # 90 degrees in radians
    st = torch.sin(theta / 2)
    ct = torch.cos(theta / 2)
    zeros = torch.zeros(horizon)
    quat_zb = torch.stack(
        [ct, zeros, zeros, st], dim=-1
    )  # Kornia is stupid, thus (w,x,y,z) changed from quat_z = torch.stack([zeros, zeros, st, ct], dim=-1)

    # Convert the quaternions to rotation objects
    R_bR_bC = kornia.geometry.quaternion_to_rotation_matrix(quat_zb).to("cuda")

    # Combine the two rotations
    R_Wt_bC = torch.matmul(R_Wt_bR, R_bR_bC).to("cuda")

    # Initial position of base_link wrt to world frame.
    base_Wt = torch.tensor([0.41614, -0.225, 1.2627], dtype=torch.float32).to("cuda")
    H_Wt_bC = torch.zeros((horizon, 4, 4), dtype=torch.float32, device="cuda")

    H_Wt_bC[:, :3, :3] = R_Wt_bC
    H_Wt_bC[:, :3, -1] = base_Wt
    H_Wt_bC[:, 3, -1] = 1.0

    R_Wt_bC_T = R_Wt_bC.transpose(1, 2)
    H_bC_Wt = torch.zeros((horizon, 4, 4), dtype=torch.float32, device="cuda")
    H_bC_Wt[:, :3, :3] = R_Wt_bC_T  # R_bC_Wt
    H_bC_Wt[:, :3, -1] = -R_Wt_bC_T @ base_Wt  # - R_bC_Wt_T @ base_Wt = base_bC
    H_bC_Wt[:, 3, -1] = 1.0

    return H_Wt_bC, H_bC_Wt


def tomm_to_coppeliaBase(pos_Wt, quat_Wt):
    ############# Important for Visualization ####################
    H_Wt_bC, H_bC_Wt = get_tomm_setup(1)

    quat_Wt = quat_Wt[[3, 0, 1, 2]]  # Kornia accepts w,x,y,z
    R_Wt_p = kornia.geometry.quaternion_to_rotation_matrix(quat_Wt)
    H_Wt_p = torch.zeros((1, 4, 4), dtype=torch.float32, device="cuda")

    H_Wt_p[:, :3, :3] = R_Wt_p
    H_Wt_p[:, :3, -1] = pos_Wt
    H_Wt_p[:, 3, -1] = 1.0

    pose_bC = H_bC_Wt @ H_Wt_p
    pos_bC = pose_bC[:, :3, -1]
    orien_bC = pose_bC[:, :3, :3].squeeze()
    quat_bC = kornia.geometry.rotation_matrix_to_quaternion(orien_bC)  # w,x,y,z
    quat_bC = quat_bC[[1, 2, 3, 0]].unsqueeze(0)

    print("goal_pos in Coppelia Sim base_link coords: ", pos_bC)
    print("goal_quaternion in Copp Sim base_link coord: ", quat_bC)

    pos_bC_numpy = pos_bC.cpu().numpy()
    quat_bC_numpy = quat_bC.cpu().numpy()
    quat_bC_numpy = quat_bC_numpy / np.linalg.norm(quat_bC_numpy)
    copp_point = np.concatenate([pos_bC_numpy, quat_bC_numpy], axis=1).squeeze()
    return copp_point


if __name__ == "__main__":
    import ipdb

    ipdb.set_trace()
