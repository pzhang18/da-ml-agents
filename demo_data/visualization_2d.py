import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm

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


# Draw the origin marker
# ax.scatter([0], [0], [0], color='red', marker='o')

'''
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show(block=True)
'''
'''
for i in range(tcp_forces_sampled.shape[1]):
    plt.plot(timestamps_sampled, tcp_forces_sampled[:,i], label=f'force {tcp_force_cols[i]}')

plt.xlabel('Time')
plt.ylabel('Force Value')
plt.legend()

'''


for i in range(tcp_forces_sampled.shape[1]):
    plt.plot(timestamps_sampled, tcp_forces_sampled[:,i], label=f'force {tcp_force_cols[i]}')

plt.xlabel('Time')
plt.ylabel('Torque Value')
plt.legend()

plt.show(block=True)
