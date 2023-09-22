import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.animation as animation

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('data/demo_data/robot_data_01.csv', delimiter=' ')

# Extract the TCP pose, for ce, torque columns from the DataFrame
tcp_pose_cols = ['actual_TCP_pose_0', 'actual_TCP_pose_1', 'actual_TCP_pose_2']
tcp_poses = df[tcp_pose_cols].values

tcp_force_cols = ['actual_TCP_force_0', 'actual_TCP_force_1', 'actual_TCP_force_2']
tcp_forces = df[tcp_force_cols].values

tcp_torque_cols = ['actual_TCP_force_3', 'actual_TCP_force_4', 'actual_TCP_force_5']
tcp_torques = df[tcp_torque_cols].values


# Select point from every 10th row
tcp_poses_sampled = tcp_poses[::10, :]
tcp_forces_sampled = tcp_forces[::10, :]
tcp_torques_sampled = tcp_torques[::10, :]

# Extract the timestamps from the DataFrame and normalize to [0,1]
timestamps = df['timestamp'].values
timestamps_sampled = timestamps[::10]
timestamps_normalized = (
    timestamps_sampled - timestamps_sampled.min()
    ) / (timestamps_sampled.max() - timestamps_sampled.min()
        )


# Generate a colormap from red to blue
# more colors here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
cmap = cm.get_cmap('cool')
# cmap = cm.get_cmap('plasma')

# Create a 3D scatter plot to visualize the TCP positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    tcp_poses_sampled[:, 0], 
    tcp_poses_sampled[:, 1], 
    tcp_poses_sampled[:, 2],
    c=timestamps_normalized,
    cmap=cmap
)
ax.plot(tcp_poses_sampled[:, 0], tcp_poses_sampled[:, 1], tcp_poses_sampled[:, 2], color = 'gray')

# Add a colorbar to show the color gradient
cbar = plt.colorbar(scatter)
cbar.set_label('Timestamp')

'''
# Plot arrows for forces
ax.quiver(
    tcp_poses_sampled[:, 0],
    tcp_poses_sampled[:, 1],
    tcp_poses_sampled[:, 2],
    tcp_forces_sampled[:, 0],
    tcp_forces_sampled[:, 1],
    tcp_forces_sampled[:, 2],
    color='green',
    length=0.001,  # Control the length of the arrows
    normalize=True  # Normalize arrow lengths
)

# Plot arrows for torques
ax.quiver(
    tcp_poses_sampled[:, 0],
    tcp_poses_sampled[:, 1],
    tcp_poses_sampled[:, 2],
    tcp_torques_sampled[:, 0],
    tcp_torques_sampled[:, 1],
    tcp_torques_sampled[:, 2],
    color='red',
    length=0.001,  # Control the length of the arrows
    normalize=True  # Normalize arrow lengths
)
'''

# Draw the origin marker
# ax.scatter([0], [0], [0], color='red', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Adjust the coordinate boundaries
# ax.set_xlim(tcp_poses_sampled[:, 0].min(), tcp_poses_sampled[:, 0].max())
# ax.set_ylim(tcp_poses_sampled[:, 1].min(), tcp_poses_sampled[:, 1].max())
# ax.set_zlim(tcp_poses_sampled[:, 2].min(), tcp_poses_sampled[:, 2].max())
ax.set_xlim(-0.17, -0.12)
ax.set_ylim(0.61,0.66)
ax.set_zlim(0.39,0.51)

plt.show(block=True)

