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
#path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\'
#your_mesh = mesh.Mesh.from_file(path + 'RD-08-L-w-FAS_2.stl')
your_mesh = mesh.Mesh.from_file('G:\\My Drive\\Project\\IntraOral Scanner Registration\\Results\\Accuracy FXT tests\\Full arch trial5\\342021-Fxt-02-342021\\342021-fxt-02-342021-lowerjaw.stl')
print('length of your_mesh points is', np.shape(your_mesh.x))


# right molar1 boundary in stl
x_Rmolar1 = [19, 31]
y_Rmolar1 = [-20, -12.7]
z_Rmolar1 = [210, 220]

# right molar2 boundary in stl
x_Rmolar2 = [18.5, 29]
y_Rmolar2 = [-11, -3]
z_Rmolar2 = [212, 219]

# right molar3 boundary in stl
x_Rmolar3 = [16.5, 26]
y_Rmolar3 = [-1.2, 7.6]
z_Rmolar3 = [212, 219]

# left molar1 boundary in stl
x_Lmolar1 = [-33, -19]
y_Lmolar1 = [-25, -16.9]
z_Lmolar1 = [210, 219]

# left molar2 boundary in stl
x_Lmolar2 = [-32, -21]
y_Lmolar2 = [-16, -7.3]
z_Lmolar2 = [210, 219]

# left molar3 boundary in stl
x_Lmolar3 = [-32, -22]
y_Lmolar3 = [-6, 3.6]
z_Lmolar3 = [211, 219]

# full arch in stl
x_full = [-15,6]
y_full = [16,32]
z_full = [212,230]



# extract molar based on boundary
points = extract_vertex_boundary(your_mesh.points, x_Rmolar1, y_Rmolar1, z_Rmolar1)
print('shape of points is', np.shape(points))

stl_point_cloud = points
stl_pc_file = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Rmolar1.csv'
#Yomiwrite.write_csv_matrix(stl_pc_file, stl_point_cloud)

pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(points)
#pcd.paint_uniform_color([0, 0.651, 0.929])
visualization.draw_geometries([pcd])


# extract molar2 based on boundary
points_R2 = extract_vertex_boundary(your_mesh.points, x_Rmolar2, y_Rmolar2, z_Rmolar2)
print('shape of points is', np.shape(points_R2))

stl_point_cloud_R2 = points_R2
stl_pc_file_R2 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Rmolar2.csv'
#Yomiwrite.write_csv_matrix(stl_pc_file_R2, stl_point_cloud_R2)

pcd_R2 = open3d.PointCloud()
pcd_R2.points = open3d.Vector3dVector(points_R2)
#pcd.paint_uniform_color([0, 0.651, 0.929])
visualization.draw_geometries([pcd_R2])


# extract molar based on boundary
points_R3 = extract_vertex_boundary(your_mesh.points, x_Rmolar3, y_Rmolar3, z_Rmolar3)
print('shape of points is', np.shape(points_R3))

stl_point_cloud_R3 = points_R3
stl_pc_file_R3 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Rmolar3.csv'
#Yomiwrite.write_csv_matrix(stl_pc_file_R3, stl_point_cloud_R3)

pcd_R3 = open3d.PointCloud()
pcd_R3.points = open3d.Vector3dVector(points_R3)
#pcd.paint_uniform_color([0, 0.651, 0.929])
visualization.draw_geometries([pcd_R3])


# extract molar based on boundary
points_L1 = extract_vertex_boundary(your_mesh.points, x_Lmolar1, y_Lmolar1, z_Lmolar1)
print('shape of points is', np.shape(points_L1))

stl_point_cloud_L1 = points_L1
stl_pc_file_L1 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Lmolar1.csv'
#Yomiwrite.write_csv_matrix(stl_pc_file_L1, stl_point_cloud_L1)

pcd_L1 = open3d.PointCloud()
pcd_L1.points = open3d.Vector3dVector(points_L1)
#pcd.paint_uniform_color([0, 0.651, 0.929])
visualization.draw_geometries([pcd_L1])


# extract molar based on boundary
points_L2 = extract_vertex_boundary(your_mesh.points, x_Lmolar2, y_Lmolar2, z_Lmolar2)
print('shape of points is', np.shape(points_L2))

stl_point_cloud_L2 = points_L2
stl_pc_file_L2= 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Lmolar2.csv'
#Yomiwrite.write_csv_matrix(stl_pc_file_L2, stl_point_cloud_L2)

pcd_L2 = open3d.PointCloud()
pcd_L2.points = open3d.Vector3dVector(points_L2)
#pcd.paint_uniform_color([0, 0.651, 0.929])
visualization.draw_geometries([pcd_L2])


# extract molar based on boundary
points_L3 = extract_vertex_boundary(your_mesh.points, x_Lmolar3, y_Lmolar3, z_Lmolar3)
print('shape of points is', np.shape(points_L3))

stl_point_cloud_L3 = points_L3
stl_pc_file_L3 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Lmolar3.csv'
#Yomiwrite.write_csv_matrix(stl_pc_file_L3, stl_point_cloud_L3)

pcd_L3 = open3d.PointCloud()
pcd_L3.points = open3d.Vector3dVector(points_L3)
#pcd.paint_uniform_color([0, 0.651, 0.929])
visualization.draw_geometries([pcd_L3])


# extract full arch based on boundary
points_full = extract_vertex_boundary(your_mesh.points, x_full, y_full, z_full)
print('shape of points is', np.shape(points_full))

stl_point_cloud_full = points_full
stl_pc_file_full = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_full.csv'
Yomiwrite.write_csv_matrix(stl_pc_file_full, stl_point_cloud_full)

pcd_full = open3d.PointCloud()
pcd_full.points = open3d.Vector3dVector(points_full)
#pcd.paint_uniform_color([0, 0.651, 0.929])
visualization.draw_geometries([pcd_full])
