import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation, Animation

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('data/demo_data/robot_data_01.csv', delimiter=' ')

# Extract the TCP pose, for ce, torque columns from the DataFrame
tcp_pose_cols = ['actual_TCP_pose_0', 'actual_TCP_pose_1', 'actual_TCP_pose_2']
tcp_poses = df[tcp_pose_cols].values

tcp_force_cols = ['actual_TCP_force_0', 'actual_TCP_force_1', 'actual_TCP_force_2']
tcp_forces = df[tcp_force_cols].values
# tcp_forces = df['actual_TCP_force_0'].values

tcp_torque_cols = ['actual_TCP_force_3', 'actual_TCP_force_4', 'actual_TCP_force_5']
tcp_torques = df[tcp_torque_cols].values

tcp_speed_cols = ['actual_TCP_speed_0', 'actual_TCP_speed_1', 'actual_TCP_speed_2', 
                  'actual_TCP_speed_3', 'actual_TCP_speed_4', 'actual_TCP_speed_5']
tcp_speeds = df[tcp_speed_cols].values


# Select point from every 10th row
tcp_poses_sampled = tcp_poses[::10, :]
tcp_forces_sampled = tcp_forces[::10, :]
tcp_torques_sampled = tcp_torques[::10, :]
tcp_speeds_sampled = tcp_speeds[::10, :]

# Calculate reward
dx = tcp_poses_sampled[:,0] - tcp_poses_sampled[-1,0]
dy = tcp_poses_sampled[:,1] - tcp_poses_sampled[-1,1]
dz = tcp_poses_sampled[:,2] - tcp_poses_sampled[-1,2]
distance = np.stack((dx, dy, dz), axis=1)
row_sums = np.sum(distance, axis=1)  # Sum along axis=1
result = row_sums[:, np.newaxis]
reward = 1 / (result + 0.001)

# select data to be visualized
data_sampled = reward
data_name_str = "Reward"

# Extract the timestamps from the DataFrame and normalize [0,1]
timestamps = df['timestamp'].values
timestamps_sampled = timestamps[::10] - timestamps.min()
timestamps_normalized = (
    timestamps_sampled - timestamps_sampled.min()
    ) / (timestamps_sampled.max() - timestamps_sampled.min()
        )

# ANIMATION: Set up the figure and axis
fig, ax = plt.subplots()

# Set the labels and limits
ax.set_xlabel('Timestamp (s)')
ax.set_ylabel(data_name_str)
ax.set_xlim(0.0, timestamps_sampled.max()-timestamps_sampled.min())  # Adjust the x-axis limits as needed
ax.set_ylim(data_sampled.min(), data_sampled.max())  # Adjust the y-axis limits as needed


# Create an empty scatter plot
# scatter = ax.scatter([], [], animated=True)

# Create empty lines for each force component
lines = [ax.plot([], [], label=data_name_str+f'{i}')[0] for i in range(data_sampled.shape[1])]
# Set the initial data for each line
for line in lines:
    line.set_data([], [])

# Animation update function
def update(frame):
    # Get the data value and timestamps up to the current frame
    data_value_frame = data_sampled[:(frame+1)]
    timestamp_frame = timestamps_sampled[:(frame+1)]

    # Update the data for each line
    for i, line in enumerate(lines):
        line.set_data(timestamp_frame, data_value_frame[:, i])

    # Update the scatter plot data
    # scatter.set_offsets(list(zip(timestamp_frame, forces_frame)))

    return lines

# Create the animation
animation = FuncAnimation(fig, update, frames=len(timestamps_sampled), interval=100, blit=True)
#animation.save('robot_data_01.mp4')

# Draw the origin marker
# ax.scatter([0], [0], [0], color='red', marker='o')

'''
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show(block=True)
'''
'''
# plot forces
for i in range(tcp_forces_sampled.shape[1]):
    plt.plot(timestamps_sampled, tcp_forces_sampled[:,i], label=f'force {tcp_force_cols[i]}')

plt.xlabel('Time')
plt.ylabel('Force Value')
plt.legend()

'''
'''
# plot torques
for i in range(tcp_torques_sampled.shape[1]):
    plt.plot(timestamps_sampled, tcp_torques_sampled[:,i], label=f'force {tcp_torque_cols[i]}')

plt.xlabel('Time')
plt.ylabel('Torque Value')
plt.legend()
'''
plt.legend()
plt.show(block=True)
# plt.show()
