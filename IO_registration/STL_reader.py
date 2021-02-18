# Test the STL read functions
# Reference can be found: https://pypi.org/project/numpy-stl/

import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from open3d import *
import open3d
import Writers as Yomiwrite


# remove duplicates from list
# https://www.geeksforgeeks.org/python-ways-to-remove-duplicates-from-list/
def extract_vertex(mesh_points):
    vertex = np.zeros(3)
    for i in range(len(mesh_points)):
        point1 = mesh_points[i][0:3]
        point2 = mesh_points[i][3:6]
        point3 = mesh_points[i][6:9]
        vertex = append_without_dup(vertex, point1)
        vertex = append_without_dup(vertex, point2)
        vertex = append_without_dup(vertex, point3)
    return vertex


def extract_vertex_boundary(mesh_points, x, y, z):
    vertex = np.zeros(3)
    for i in range(len(mesh_points)):
        point1 = mesh_points[i][0:3]
        point2 = mesh_points[i][3:6]
        point3 = mesh_points[i][6:9]
        vertex = append_point_in(vertex, point1, x, y, z)
        vertex = append_point_in(vertex, point2, x, y, z)
        vertex = append_point_in(vertex, point3, x, y, z)
    return vertex[1:,:]


def append_point_in(vertex, point, x, y, z):
    if x[0] < point[0] < x[1]:
        if y[0] < point[1] < y[1]:
            if z[0] < point[2] < z[1]:
                vertex = append_without_dup(vertex, point)
    return vertex


def append_without_dup(array, element):
    if element not in array:
        array = np.vstack((array, element))
    return array


# using existing stl files
path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\'
your_mesh = mesh.Mesh.from_file(path + 'RD-08-L-w-FAS_2.stl')
print('length of your_mesh points is', np.shape(your_mesh.x))


# right molar boundary in stl
x_boundary = [5, 19]
y_boundary = [42, 50]
z_boundary = [-41, -25]

# right premolar boundary in stl
premolar_x = [-9, -2.5]
premolar_y = [44, 50]
premolar_z = [-33, -21]

# extract molar based on boundary
points = extract_vertex_boundary(your_mesh.points, x_boundary, y_boundary, z_boundary)
print('shape of points is', np.shape(points))

stl_point_cloud = points
stl_pc_file = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points.csv'
Yomiwrite.write_csv_matrix(stl_pc_file, stl_point_cloud)

pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(points)
#pcd.paint_uniform_color([0, 0.651, 0.929])
visualization.draw_geometries([pcd])

# extract premolar based on boundary
pre_molar_points = extract_vertex_boundary(your_mesh.points, premolar_x, premolar_y, premolar_z)
stl_point_cloud_2teeth = np.vstack((points, pre_molar_points))
stl_pc_file_2teeth = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_2teeth.csv'
Yomiwrite.write_csv_matrix(stl_pc_file_2teeth, stl_point_cloud_2teeth)
pcd_pre = open3d.PointCloud()
pcd_pre.points = open3d.Vector3dVector(pre_molar_points)
visualization.draw_geometries([pcd_pre])



