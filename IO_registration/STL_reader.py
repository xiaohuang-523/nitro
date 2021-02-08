# Test the STL read functions
# Reference can be found: https://pypi.org/project/numpy-stl/

import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import pyplot


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
    print('array is', array)
    print('element is', element)
    if element not in array:
        array = np.concatenate((array, element), axis=0)
    return array


# using existing stl files
path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\'
your_mesh = mesh.Mesh.from_file(path + 'Post_splint.stl')
#print('mesh vector is', your_mesh.vectors)
#print('mesh point is', your_mesh.points)
print('point shape is', your_mesh.x[0])
print('point shape is', your_mesh.x[1])
print('point shape is', your_mesh.x[2])

points = extract_vertex(your_mesh.points[0:3])
print('shape of points is', np.shape(points))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(your_mesh.x[0:3],your_mesh.y[0:3],your_mesh.z[0:3], c='r', s=4)

plt.show()


# Or creating a new mesh (make sure not to overwrite the `mesh` import by
# naming it `mesh`):
#VERTICE_COUNT = 100
#data = numpy.zeros(VERTICE_COUNT, dtype=mesh.Mesh.dtype)
#your_mesh = mesh.Mesh(data, remove_empty_areas=False)


# Create a new plot
#figure = pyplot.figure()
#axes = mplot3d.Axes3D(figure)

# Load the STL files and add the vectors to the plot
#axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
#scale = your_mesh.points.flatten()
#axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
#pyplot.show()