import h5py
import numpy as np
import pandas as pd
import math

# from mlagents.trainers.torch_entities.distributions import GaussianDistInstance

# Adding human demo data to the replay buffer. 
# Opens the h5py file saved by training process. 
# Loads csv files recorded on real robot
# Overwrite corresponding keys in the h5py file

buffer_file = "C:/Users/pengf/rl/results/000/AgentTCP/test.hdf5"
demo_path = "C:/Users/pengf/rl/da-ml-agents/demo_data/human_demo/0905_robot_data_00.csv"

def load_demo(path):
    data_dict = {}
    df_raw = pd.read_csv(path, delimiter=' ')
    final_state = df_raw.iloc[-1]
    # Extract the TCP pose, for ce, torque columns from the DataFrame
    # UR robot coordinate XYZ needs to convert to Unity's XZY coordinate
    tcp_position_cols = ['actual_TCP_pose_0', 'actual_TCP_pose_2', 'actual_TCP_pose_1']
    tcp_rotation_cols = ['actual_TCP_pose_3', 'actual_TCP_pose_5', 'actual_TCP_pose_4']
    # rx, ry and rz is a rotation vector representation of the tool orientation
    tcp_force_cols = ['actual_TCP_force_0', 'actual_TCP_force_2', 'actual_TCP_force_1']
    tcp_torque_cols = ['actual_TCP_force_3', 'actual_TCP_force_5', 'actual_TCP_force_4']
    tcp_speed_cols = ['actual_TCP_speed_0', 'actual_TCP_speed_2', 'actual_TCP_speed_1', 
                    'actual_TCP_speed_3', 'actual_TCP_speed_5', 'actual_TCP_speed_4']
    # remove uncecessary data recorded after reaching the taget
    df_filtered = df_raw[(abs(df_raw['actual_TCP_pose_0']+df_raw['actual_TCP_pose_1']+df_raw['actual_TCP_pose_2']
                       -final_state['actual_TCP_pose_0']-final_state['actual_TCP_pose_1']
                       -final_state['actual_TCP_pose_2'])>0.0015)]
    # match data frequency with unity time step
    data_f = 100 # data recorded at 100hz frequency
    unity_f = 50 # unity default 50hz frequency (0.02s time step)
    df = df_filtered.iloc[::int(data_f / unity_f)]
    # get values of each category 
    data_length = df.shape[0]
    target_position = np.tile(final_state[tcp_position_cols].values, (data_length,1))
    target_rotation = to_rotation(np.tile(final_state[tcp_rotation_cols].values,(data_length,1)))
    tcp_position = df[tcp_position_cols].values 
    tcp_rotation = to_rotation(df[tcp_rotation_cols].values) 
    tcp_forces = df[tcp_force_cols].values
    tcp_torques = df[tcp_torque_cols].values
    tcp_speeds = df[tcp_speed_cols].values # unity anglear velocity in rad/s, so no need to convert
    reward = get_reward(tcp_position, tcp_rotation, target_position, target_rotation)

    # observation (24 values) order: 
    # target_position, tcp_position, tcp_rotation, tcp_forces, tcp_torques, tcp_speeds
    obs = np.hstack((target_position,target_rotation, tcp_position, tcp_rotation,
                    tcp_forces, tcp_torques, tcp_speeds))
    next_obs = np.vstack((obs[1:], obs[-1]))
    continuous_action = tcp_speeds
    continuous_log_probs = np.empty((data_length,0))
    next_continuous_action = np.vstack((tcp_speeds[1:], tcp_speeds[-1]))
    done = np.zeros(data_length)
    done[-1] = 1
    environment_rewards = reward
    # create data dict
    data_dict["obs:0"] = obs.astype(np.float32)
    data_dict["next_obs:0"] = next_obs.astype(np.float32)
    data_dict["continuous_action"] = continuous_action.astype(np.float32)
    data_dict["continuous_log_probs"] = continuous_log_probs.astype(np.float32)
    data_dict["next_continuous_action"] = next_continuous_action.astype(np.float32)
    data_dict["done"] = done.astype(np.float32)
    data_dict["environment_rewards"] = environment_rewards.astype(np.float32)

    return data_dict

def get_reward(p, r, target_p, target_r):
    distance = np.sqrt(np.sum((p-target_p)**2, axis=1))
    angleDelta = np.sum(abs(r-target_r), axis=1)
    reward = 1.0 - (distance + angleDelta*0.001)
    mask = reward > (1.0 - 0.01)
    reward[mask] += 9.0
    return reward

def to_rotation(vector):
    rx = vector[:, 0]
    ry = vector[:, 1]
    rz = vector[:, 2]
    rotation_x_deg = np.degrees(np.arctan2(ry, rz)).reshape(-1, 1)
    rotation_y_deg = np.degrees(np.arctan2(rx, rz)).reshape(-1, 1)
    rotation_z_deg = np.degrees(np.arctan2(ry, rx)).reshape(-1, 1)
    rotation = np.hstack((rotation_x_deg, rotation_y_deg, rotation_z_deg))
    return rotation

def run():
    with h5py.File(buffer_file, 'r') as file:
        buffer_dict = {}  # Create a dictionary to store the data
        
        # Iterate through the keys in the HDF5 file
        for key in file.keys():
            # Load the dataset associated with the key
            data_value = file[key][:]
            print(key, data_value[:5])
            # Add the dataset to the dictionary with the key as the name
            buffer_dict[key] = data_value

    # Export the DataFrame to a single CSV file
    # df.to_csv('output.csv', index=False)
    demo_data = load_demo(demo_path)

    # Open the HDF5 file in read-write mode ('a')
    with h5py.File(buffer_file, 'a') as file:
        # Navigate to the specific dataset or group you want to modify
        dataset = file['dataset_name']  # Replace 'dataset_name' with the actual dataset name
        
        # Modify the values in the dataset
        new_values = ...  # Replace with the new values you want to assign
        dataset[:] = new_values

def main():
    run()

if __name__ == "__main__":
    main()



    

# to modify values
# Open the HDF5 file in read-write mode ('a')
# with h5py.File('your_file.h5', 'a') as file:
#     # Navigate to the specific dataset or group you want to modify
#     dataset = file['dataset_name']  # Replace 'dataset_name' with the actual dataset name
    
#     # Modify the values in the dataset
#     new_values = ...  # Replace with the new values you want to assign
#     dataset[:] = new_values
    
# The changes will be saved to the HDF5 file when the file is closed