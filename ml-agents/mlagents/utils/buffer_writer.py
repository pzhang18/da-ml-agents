import h5py
import numpy as np
import pandas as pd

file_object = "C:/Users/pengf/rl/results/000/AgentTCP/test.hdf5"
with h5py.File(file_object, "r") as read_file:
    for key in read_file.keys():
        print(key, read_file[key])
        #print(read_file[key][1])
    print(read_file["obs:0"][100])

with h5py.File(file_object, 'r') as file:
    data_dict = {}  # Create a dictionary to store the data
    
    # Iterate through the keys in the HDF5 file
    for key in file.keys():
        # Load the dataset associated with the key
        dataset = file[key][:]
        
        # Add the dataset to the dictionary with the key as the name
        data_dict[key] = dataset.tolist()

    # Create a DataFrame from the data dictionary
    # df = pd.DataFrame(data_dict)

    # Export the DataFrame to a single CSV file
    # df.to_csv('output.csv', index=False)
    

# to modify values
# Open the HDF5 file in read-write mode ('a')
with h5py.File('your_file.h5', 'a') as file:
    # Navigate to the specific dataset or group you want to modify
    dataset = file['dataset_name']  # Replace 'dataset_name' with the actual dataset name
    
    # Modify the values in the dataset
    new_values = ...  # Replace with the new values you want to assign
    dataset[:] = new_values
    
# The changes will be saved to the HDF5 file when the file is closed