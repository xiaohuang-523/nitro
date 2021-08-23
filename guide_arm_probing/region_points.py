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
import plot as Yomiplot
import point_cloud_manipulation as pcm


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

def append_without_dup(array, element):
    if element not in array:
        array = np.vstack((array, element))
    return array


def read_5_regions():
    # using existing stl files
    path = 'G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\Edentulous registration error analysis\\Typodont_scan\\regions_5\\'
    all_vertex = []

    for i in range(5):
        your_mesh = mesh.Mesh.from_file(path + '5_' + np.str(i+1) + '.stl')
        vertex = extract_vertex(your_mesh.points)
        sub_vertex = np.asarray(pcm.preprocess_point_cloud(vertex, 2.0))
        print('shape of sub_vertex is', np.shape(sub_vertex))
        all_vertex.append(sub_vertex)
    return all_vertex


def read_4_regions():
    # using existing stl files
    path = 'G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\Edentulous registration error analysis\\Typodont_scan\\regions_4\\'
    all_vertex = []

    for i in range(4):
        your_mesh = mesh.Mesh.from_file(path + '4_' + np.str(i+1) + '.stl')
        vertex = extract_vertex(your_mesh.points)
        sub_vertex = np.asarray(pcm.preprocess_point_cloud(vertex, 2.0))
        print('shape of sub_vertex is', np.shape(sub_vertex))
        all_vertex.append(sub_vertex)
    return all_vertex


def read_3_regions():
    # using existing stl files
    path = 'G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\Edentulous registration error analysis\\Typodont_scan\\regions_3\\'
    all_vertex = []

    for i in range(3):
        your_mesh = mesh.Mesh.from_file(path + '3_' + np.str(i+1) + '.stl')
        vertex = extract_vertex(your_mesh.points)
        sub_vertex = np.asarray(pcm.preprocess_point_cloud(vertex, 2.0))
        print('shape of sub_vertex is', np.shape(sub_vertex))
        all_vertex.append(sub_vertex)
    return all_vertex


def read_all():
    path = 'G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\Edentulous registration error analysis\\\Typodont_scan\\Tooth surface.stl'
    your_mesh = mesh.Mesh.from_file(path)
    print('read stl file')
    print('total number of points', len(your_mesh.points))
    vertex = extract_vertex(your_mesh.points)
    print('extract vertex')
    sub_vertex = np.asarray(pcm.preprocess_point_cloud(vertex, 1.0))
    print('shape of sub_vertex is', np.shape(sub_vertex))
    return sub_vertex


def read_regions(file, voxel_size):
    your_mesh = mesh.Mesh.from_file(file)
    vertex = extract_vertex(your_mesh.points)
    sub_vertex = np.asarray(pcm.preprocess_point_cloud(vertex, voxel_size))
    print('shape of sub_vertex is', np.shape(sub_vertex))
    return sub_vertex


# print('shape of all vertex is', np.shape(all_vertex[0]))
# fig1 = plt.figure()
# ax = fig1.add_subplot(111, projection='3d')
# Yomiplot.plot_3d_points(all_vertex[0], ax, color = 'r')
# Yomiplot.plot_3d_points(all_vertex[1], ax, color = 'g')
# Yomiplot.plot_3d_points(all_vertex[2], ax, color = 'r')
# Yomiplot.plot_3d_points(all_vertex[3], ax, color = 'g')
# Yomiplot.plot_3d_points(all_vertex[4], ax, color = 'r')
# plt.show()