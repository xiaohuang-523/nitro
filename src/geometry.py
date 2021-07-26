import numpy as np
from shapely.geometry import Point, Polygon


class plane3D:
    def __init__(self, o, z):
        self.origin = o
        #self.x_axis = x
        #self.y_axis = y
        self.z_axis = z


def plane_intersection(plane1, plane2, plane3):
    p1 = plane1.origin
    print('p1 is', p1)

    p2 = plane2.origin
    print('p2 is', p2)
    p3 = plane3.origin
    z1 = plane1.z_axis
    z2 = plane2.z_axis
    z3 = plane3.z_axis
    A = np.asarray([z1, z2, z3])
    # A = np.array([[p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]],
    #               [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]],
    #               [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
    #               ])
    # B = np.array([p1[0]*p3[0] - p1[0]*p2[0] + p1[1]*p3[1] - p1[1]*p2[1] + p1[2]*p3[2] - p1[2]*p2[2],
    #               p1[0]*p2[0] - p2[0]*p3[0] + p1[1]*p2[1] - p2[1]*p3[1] + p1[2]*p2[2] - p2[2]*p3[2],
    #               p1[0]*p3[0] - p2[0]*p3[0] + p1[1]*p3[1] - p2[1]*p3[1] + p1[2]*p3[2] - p2[2]*p3[2]])
    B = np.asarray([np.matmul(z1, p1), np.matmul(z2, p2), np.matmul(z3, p3)])
    # check singularity
    rank = np.linalg.matrix_rank(A)
    print('A is', A)
    print('rank is', rank)
    if rank == 3:
        x = np.matmul(np.linalg.inv(A), B)
    else:
        raise ValueError('matrix is singular, no intersection point among the selected three planes')
    #x = np.matmul(np.linalg.pinv(A), B)
    return x


# Useful info: https://automating-gis-processes.github.io/CSC18/lessons/L4/point-in-polygon.html
# Inputs:
#       1. point to check
#       2. list of polygon corner points
def point_in_polygon(point, corner_points_list):
    # create a polygon
    p1 = Point(point)
    poly = Polygon(corner_points_list)
    return p1.within(poly)



