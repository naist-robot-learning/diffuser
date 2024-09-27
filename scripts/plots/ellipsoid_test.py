import pybullet as p
import pybullet_data
import numpy as np
import time
import torch
from diffuser.robot.UR5kinematicsAndDynamics_vectorized import compute_kinetic_energy_matrix
import einops


def compute_traj_eigendecomposition(theta):

    q_ = torch.tensor(theta).unsqueeze(dim=0)
    q_ = einops.rearrange(q_, "B H T -> B T H")

    M_x_inv = compute_kinetic_energy_matrix(q_)
    L, V_ = torch.linalg.eig(M_x_inv)
    L = 1 / L.real.cpu()
    L = torch.sqrt(L) * 0.15
    V = V_.real.cpu().numpy()
    return L, V


# Function to initialize the ellipsoid lines
def initialize_ellipsoid_lines(n_lines=60):
    line_ids = []
    for _ in range(n_lines * (n_lines // 2)):
        line_id = p.addUserDebugLine([0, 0, 0], [0, 0, 0], lineColorRGB=[1, 0, 0], lineWidth=1)
        line_ids.append(line_id)
    return line_ids


# Function to update the ellipsoid lines
def update_ellipsoid_lines(center, radii, rotation_matrix, line_ids, n_lines=60):
    line_index = 0
    for theta in np.linspace(0, 2 * np.pi, n_lines):
        for phi in np.linspace(0, np.pi, n_lines // 2):
            # Parametric equation of an ellipsoid in 3D
            x = radii[0] * np.sin(phi) * np.cos(theta)
            y = radii[1] * np.sin(phi) * np.sin(theta)
            z = radii[2] * np.cos(phi)

            # Apply the rotation and translation
            point = np.dot(rotation_matrix, np.array([x, y, z])) + center

            # Next point along the line for drawing
            next_phi = phi + np.pi / n_lines
            next_theta = theta + 2 * np.pi / n_lines

            x_next = radii[0] * np.sin(next_phi) * np.cos(next_theta)
            y_next = radii[1] * np.sin(next_phi) * np.sin(next_theta)
            z_next = radii[2] * np.cos(next_phi)

            next_point = np.dot(rotation_matrix, np.array([x_next, y_next, z_next])) + center

            # Update line segment between the points
            p.addUserDebugLine(
                point, next_point, lineColorRGB=[1, 0, 0], lineWidth=1, replaceItemUniqueId=line_ids[line_index]
            )
            line_index += 1


# Start PyBullet simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())


# Load the UR5 robot with a fixed base
ur5_id = p.loadURDF("scripts/ur5_ellip.urdf", useFixedBase=True)

# Create a large white plane to act as the background
plane_id = p.loadURDF("plane.urdf", [0, 0, -10], globalScaling=50)  # Large scaling and far away
p.changeVisualShape(plane_id, -1, rgbaColor=[1, 1, 1, 1])  # Set plane color to white


colors = {
    "base": [0.1, 0.1, 0.8, 1.0],  # Blue color
    "shoulder_link": [0.8, 0.8, 0.8, 1.0],  # Light gray
    "upper_arm_link": [0.8, 0.8, 0.8, 1.0],  # Light gray
    "forearm_link": [0.8, 0.8, 0.8, 1.0],  # Light gray
    "wrist_1_link": [0.8, 0.8, 0.8, 1.0],  # Light gray
    "wrist_2_link": [0.8, 0.8, 0.8, 1.0],  # Light gray
    "wrist_3_link": [0.8, 0.8, 0.8, 1.0],  # Light gray
}


# Apply colors to each link
for i in range(p.getNumJoints(ur5_id)):
    joint_info = p.getJointInfo(ur5_id, i)
    link_name = joint_info[12].decode("utf-8")  # Link name

    if link_name in colors:
        p.changeVisualShape(ur5_id, i, rgbaColor=colors[link_name])

p.changeVisualShape(ur5_id, -1, rgbaColor=[0.1, 0.1, 0.8, 1.0])  # Base link color


p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Enable shadows to visualize light direction


# Set the camera to simulate light coming from -x, -y direction
cameraDistance = 1.5
cameraYaw = -135  # Rotate the camera to -135 degrees to simulate light from -x, -y
cameraPitch = -40
cameraTargetPosition = [0.5, 0, 0]

p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)


# Create sliders for each joint
joint_sliders = []
for i in range(7):
    joint_sliders.append(p.addUserDebugParameter(f"Joint {i+1}", -np.pi, np.pi, 0))

# Initialize the ellipsoid lines
line_ids = initialize_ellipsoid_lines()
# Add a toggle for automatic/manual update
update_mode_toggle = p.addUserDebugParameter("Auto Update", 0, 1, 0)  # 1 for Auto, 0 for Manual
manual_update_button = p.addUserDebugParameter("Manual Update", 1, 0, 0)  # Button for manual update

# Main simulation loop
while True:
    # Read slider values to set joint angles
    joint_angles = [p.readUserDebugParameter(slider) for slider in joint_sliders]

    # Set the joint positions
    for i in range(len(joint_angles)):
        p.resetJointState(ur5_id, i, joint_angles[i])

    # Check if automatic update is enabled
    auto_update = p.readUserDebugParameter(update_mode_toggle) == 1

    # Check if manual update button is pressed
    manual_update = p.readUserDebugParameter(manual_update_button)

    if auto_update or manual_update:
        del joint_angles[0]
        q_torch = torch.tensor(joint_angles).unsqueeze(0).to("cuda")
        L, V = compute_traj_eigendecomposition(q_torch)

        # Get the position of the UR5's end-effector
        ee_link_index = 7  # The index of the end-effector link in the UR5
        ee_state = p.getLinkState(ur5_id, ee_link_index)
        ee_position = ee_state[0]  # Position of the end-effector

        # Radii of the ellipsoid correspond to the square roots of the eigenvalues
        radii = L[0]

        # Update the ellipsoid lines based on the current configuration
        update_ellipsoid_lines(ee_position, radii, V[0], line_ids)

        # Step simulation
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
