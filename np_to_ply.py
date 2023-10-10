
import open3d
import sys
import numpy as np
import random
import os  
import math as m

max_val = np.array([[157.44165455, 174.82071387, 179.34331285]])
min_val = np.array([[-160.78322951, -189.21882333, -167.02990417]])
max_min_diff = np.array([[318.22488406, 364.03953721, 346.37321702]])
mean = np.array([[0.50736506, 0.52182236, 0.46978139]]) 
std = np.array([[0.06373268, 0.06003646, 0.08100745]])

source_path = '/home/tasnim/from_004/Point-Transformers/predictions/output_00000.npy'
dest_path = '/home/tasnim/from_004/Point-Transformers/np_to_ply_processed/dest_ply'

np_load_arr = np.load(source_path) # (4096,2,3)
print(np_load_arr.shape)

surface = np_load_arr[:,0] # (4096,3)
center = np_load_arr[:,1] # (4096,3)

surface = surface * std + mean
surface = surface * max_min_diff + min_val

center = center * std + mean
center = center * max_min_diff + min_val

# surface = np.expand_dims(surface, axis=0) # (1,4096,3)
# center = np.expand_dims(center, axis=0) # (1,4096,3)


pipe_and_axis_pcd = open3d.geometry.PointCloud() # create a pointcloud
pipe_and_axis_pcd.points = open3d.utility.Vector3dVector(np.concatenate((surface,center), axis = 0)) # convert np array to point cloud
#pipe_and_axis_pcd.translate((tx, ty, tz)) # augmentation - translation
#pipe_and_axis_pcd.rotate(pipe_and_axis_pcd.get_rotation_matrix_from_xyz((rx, ry, rz)),center= pipe_and_axis_pcd.get_center()) # aumentation - rotation
pipe_and_axis_instance = 'point_cloud_pipe_and_axis.ply'
open3d.io.write_point_cloud(os.path.join(dest_path, pipe_and_axis_instance), pipe_and_axis_pcd)

#save numpy points

# print('dataset generation progress ', idx*100/dataset_size, '%', end = '\r')