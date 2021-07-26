# Library for 3d point cloud manipualtion
# Xiao Huang @ 7/6/2021


# for point subsampling
import open3d as o3d
import numpy as np
import copy
def preprocess_point_cloud(points, voxel_size):
    source = o3d.PointCloud()
    source.points = o3d.Vector3dVector(points)
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = o3d.geometry.voxel_down_sample(source, voxel_size)
    return pcd_down.points

