import sys
import numpy as np
import open3d 
import open3d.ml.torch as ml3d
import random
import os  
import torch
import math as m

# change all paths
# source_folder = '/home/tasnim/pipe_center_prediction/dgcnn.pytorch/dgcnn_test_pipe_predictions'
temp_load_path = '/home/tasnim/Open3D-ML/dataset_uniform_train/numpy_all_points/numpy_pipe_and_axis_00000.npy'
temp_save_path = '/home/tasnim/Open3D-ML/gaussian_noise_one'

# original = np.array([[1, 2, 3],
#                      [100, 200, 300],
#                      [10, 20, 30]])
# noise = np.random.normal(0, .1, original.shape)
# new_signal = original + noise
# print(new_signal)

np_load_arr = np.load(temp_load_path) # (4096,2,3)
surface = np_load_arr[:,0] # (4096,3)
center = np_load_arr[:,1] # (4096,3)

surface = np.expand_dims(surface, axis=0) # (1,4096,3)
center = np.expand_dims(center, axis=0) # (1,4096,3)

# adding noise
noise = np.random.normal(0, 0.5, surface.shape)
new_surface = surface + noise

surface = np.squeeze(new_surface, axis = 0)
center = np.squeeze(center, axis = 0)

cylinder_surface_and_corresponding_axis_points = np.stack((surface, center), axis = 1)
print(np.shape(cylinder_surface_and_corresponding_axis_points[:,0]))

# save pipe and axis point cloud
pipe_and_axis_pcd = open3d.geometry.PointCloud() # create a pointcloud
pipe_and_axis_pcd.points = open3d.utility.Vector3dVector(np.concatenate((cylinder_surface_and_corresponding_axis_points[:,0],cylinder_surface_and_corresponding_axis_points[:,1]), axis = 0)) # convert np array to point cloud
#pipe_and_axis_pcd.translate((tx, ty, tz)) # augmentation - translation
#pipe_and_axis_pcd.rotate(pipe_and_axis_pcd.get_rotation_matrix_from_xyz((rx, ry, rz)),center= pipe_and_axis_pcd.get_center()) # aumentation - rotation
pipe_and_axis_instance = 'point_cloud_pipe_and_axis.ply'
open3d.io.write_point_cloud(os.path.join(temp_save_path, pipe_and_axis_instance), pipe_and_axis_pcd)

#save numpy points
numpy_filename = 'numpy_pipe_and_axis.npy'
np.save(os.path.join(temp_save_path, numpy_filename), cylinder_surface_and_corresponding_axis_points) 

# print('dataset generation progress ', idx*100/dataset_size, '%', end = '\r')


