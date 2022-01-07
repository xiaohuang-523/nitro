import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from open3d import *
import open3d
import Writers as Yomiwrite
import define_box


def trim_tooth_box(mesh, output_file):
    your_mesh = mesh
    points = extract_vertex_boundary_box(your_mesh.points)
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(points)
    visualization.draw_geometries([pcd])

    stl_point_cloud = points
    stl_pc_file = output_file
    Yomiwrite.write_csv_matrix(stl_pc_file, stl_point_cloud)


def extract_vertex_boundary_box(mesh_points):
    vertex = np.zeros(3)
    for i in range(len(mesh_points)):
        point1 = mesh_points[i][0:3]
        point2 = mesh_points[i][3:6]
        point3 = mesh_points[i][6:9]
        vertex = append_without_dup(vertex, point1)
        vertex = append_without_dup(vertex, point2)
        vertex = append_without_dup(vertex, point3)
    return vertex[1:,:]


def append_without_dup(array, element):
    if element not in array:
        array = np.vstack((array, element))
    return array

FILE_BASE = "C:\\tools probing cadaver lab\\IOS_segmentation\\"
STL_BASE = "slt_files\\"
RESULT_BASE = "stl_segmentation\\"
your_mesh = mesh.Mesh.from_file(FILE_BASE + STL_BASE + "example.stl")
tooth_number = 17
result = FILE_BASE + RESULT_BASE + "new_stl_points_tooth" + np.str(tooth_number) + '.csv'
print('length of your_mesh points is', np.shape(your_mesh.x))
trim_tooth_box(your_mesh, result)