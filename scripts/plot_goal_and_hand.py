import sys
import numpy as np
sys.path.insert(0, "/home/ws/src")
from CoppeliaEnv4Diffuser.gymEnvironments import CoppeliaGym, CoppeliaGymFull
env = CoppeliaGymFull()
obs, goal, hand = env.reset(state_type='path')
import matplotlib.pyplot as plt
fig = plt.figure(facecolor='none', figsize=(14,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(goal[0], goal[1], goal[2], color='r')
ax.scatter(hand[0], hand[1], hand[2], color='b')

# Setting labels and aspect ratio
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1,1,1])
ticks = np.linspace(-0.5, 0.5, num=5)
ticksz = np.linspace(-0.5, 1.0, num=7)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticksz)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 1.0)

plt.show()