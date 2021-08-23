# Xiao Huang @ 07/02/2021
# This script is used to generate random points in 3d space around a given points.
# The direction of the error is isotropic and the radius (distance error) follows the normal distribution.
# The original post is:
# https://stackoverflow.com/questions/22439193/how-to-generate-new-points-as-offset-with-gaussian-distribution-for-some-points
# Uniformly distributed points on sphere: https://mathworld.wolfram.com/SpherePointPicking.html

import numpy as np
import math as m
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def spherical_to_cartesian(vec):
    """
    Convert spherical polar coordinates to cartesian coordinates:

    See the definition of spherical_cartesian_to_polar.

    @param vec: A vector of the 3 polar coordinates (r, u, v)
    @return: (x, y, z)
    """
    (r, u, v) = vec
    x = r * m.sin(u) * m.cos(v)
    y = r * m.sin(u) * m.sin(v)
    #print('y is', y)
    z = r * m.cos(u)
    return [x, y, z]


def random_3d_points(num_points, radius_mean, radius_std):
    U = np.random.random(num_points)
    V = np.random.random(num_points)
    radius = np.random.normal(radius_mean, radius_std, num_points)
    points = np.array([spherical_to_cartesian([r, 2 * np.pi * u, np.arccos(2*v - 1)]) for r,u,v in zip(radius, U,V)])
    return points


# Calculate the transformation matrix due to the orientation of the robot end-effector (probing)
# details check https://neocis.atlassian.net/browse/HW-9236
# for 2d points
def biased_transformation(point_M, point_O):
    alpha = np.arctan2(-(point_M[0] - point_O[0]), point_M[1] - point_O[1])
    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha), np.cos(alpha)]])
    t = point_M - point_O
    T = np.eye(3)
    T[0:2, 0:2] = R[:]
    T[0:2, 2] = t[0:2]
    return T


# Adding biased errors on fiducial measurements. For details, check https://neocis.atlassian.net/browse/HW-9236
# Orientation 2
def biased_3d_points_2(old_fiducials, center_point, radius_mean, radius_std):
    fiducials = np.copy(old_fiducials)
    U = np.random.random(1)     # phi
    V = np.random.random(1)     # theta
    radius = np.random.normal(radius_mean, radius_std, 1)
    [x,y,z] = np.array(spherical_to_cartesian([radius, 2 * np.pi * U, np.arccos(2*V - 1)]))

    # calculate the transformation matrix due to the offset of center point in world space
    T_offset = np.eye(3)
    offset = center_point
    T_offset[0:2, 2] = offset[0:2]
    new_fiducial = []
    for fiducial in fiducials:
        # calculate transformation from S_plane to world space (O)
        T = biased_transformation(fiducial, center_point)

        # calculate S_plane coordinates based on spherical coordinate
        x_s = y
        y_s = z

        # convert S_plane to world space
        vector = np.ones(3)
        vector[0] = x_s
        vector[1] = y_s
        new_point = np.matmul(T, vector)
        new_point = np.matmul(T_offset, new_point) # convert to the original world space if there is any offset

        z_final = fiducial[2] + x
        y_final = new_point[1]
        x_final = new_point[0]

        fiducial_tem = np.ones(3)
        fiducial_tem[0] = x_final
        fiducial_tem[1] = y_final
        fiducial_tem[2] = z_final

        new_fiducial.append(fiducial_tem)
    #plot_check_error(new_fiducial, fiducials)
    return np.asarray(new_fiducial)


# Adding biased errors on fiducial measurements. For details, check https://neocis.atlassian.net/browse/HW-9236
# Orientation 1
def biased_3d_points_1(old_fiducials, center_point, radius_mean, radius_std):
    fiducials = np.copy(old_fiducials)
    U = np.random.random(1)     # phi
    V = np.random.random(1)     # theta
    radius = np.random.normal(radius_mean, radius_std, 1)
    [x,y,z] = np.array(spherical_to_cartesian([radius, 2 * np.pi * U, np.arccos(2*V - 1)]))

    # calculate the transformation matrix due to the offset of center point in world space
    T_offset = np.eye(3)
    offset = center_point
    T_offset[0:2, 2] = offset[0:2]
    new_fiducial = []
    for fiducial in fiducials:
        # add the same biased error to all fiducials

        z_final = fiducial[2] + z
        y_final = fiducial[1] + y
        x_final = fiducial[0] + x

        fiducial_tem = np.ones(3)
        fiducial_tem[0] = x_final
        fiducial_tem[1] = y_final
        fiducial_tem[2] = z_final

        new_fiducial.append(fiducial_tem)
    #plot_check_error(new_fiducial, fiducials)
    return np.asarray(new_fiducial)


# Adding biased errors on fiducial measurements. For details, check https://neocis.atlassian.net/browse/HW-9236
# Orientation 3
def biased_3d_points_3(old_fiducials, center_point, radius_mean, radius_std):
    fiducials = np.copy(old_fiducials)
    U = np.random.random(1)     # phi
    V = np.random.random(1)     # theta
    radius = np.random.normal(radius_mean, radius_std, 1)
    #[x,y,z] = np.array(spherical_to_cartesian([radius, 2 * np.pi * U, np.arccos(2*V - 1)]))

    # calculate the transformation matrix due to the offset of center point in world space
    T_offset = np.eye(3)
    offset = center_point
    T_offset[0:2, 2] = offset[0:2]
    new_fiducial = []
    for fiducial in fiducials:
        # check if the fiducial point is in the left hand side (robot is in elbow left configuration)
        if fiducial[0] < 0:
            [x, y, z] = np.array(spherical_to_cartesian([radius, 2 * np.pi * U, np.arccos(2 * V - 1)]))
        # if fiducial is in the right hand side of the arch (robot is in elbow right configuration)
        else:
            [x, y, z] = np.array(spherical_to_cartesian([radius, 2 * np.pi * U, -np.arccos(2 * V - 1)]))
        # calculate transformation from S_plane to world space (O)
        T = biased_transformation(fiducial, center_point)

        # calculate S_plane coordinates based on spherical coordinate
        x_s = y
        y_s = z

        # convert S_plane to world space
        vector = np.ones(3)
        vector[0] = x_s
        vector[1] = y_s
        new_point = np.matmul(T, vector)
        new_point = np.matmul(T_offset, new_point) # convert to the original world space if there is any offset

        z_final = fiducial[2] + x
        y_final = new_point[1]
        x_final = new_point[0]

        fiducial_tem = np.ones(3)
        fiducial_tem[0] = x_final
        fiducial_tem[1] = y_final
        fiducial_tem[2] = z_final

        new_fiducial.append(fiducial_tem)
    #plot_check_error(new_fiducial, fiducials)
    return np.asarray(new_fiducial)


def plot_check_error(new_points, old_points):
    errors = np.asarray(new_points) - np.asarray(old_points)
    unit_errors = []
    error_norms = []
    for error in errors:
        # normalize error vectors
        error_norm = np.linalg.norm(error)
        unit_error = error / error_norm
        error_norms.append(error_norm)
        unit_errors.append(unit_error)

    unit_errors = np.asarray(unit_errors)
    #error_norms = np.asarray(error_norms)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(unit_errors)):
        print('unit errors are', unit_errors)
        ax.quiver(unit_errors[i, 0]/2, unit_errors[i, 1]/2, unit_errors[i, 2]/2, unit_errors[i,0], unit_errors[i,1], unit_errors[i,2], length=0.5, pivot='tail')
        #, pivot='tail', length=0.05, arrow_length_ratio=0.2, color='blue')

    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=-1, top=1)

    # Create axes labels
    ax.set_xlabel('X(mm)')
    ax.set_ylabel('Y(mm)')
    ax.set_zlabel('Z(mm)')
    plt.show()


    fig2 = plt.figure()
    plt.scatter(range(len(error_norms)), error_norms)
    plt.show()


def check_code(n = 500, mean = 1, std = 0.15):
    fig, ax = plt.subplots()
    ax = Axes3D(fig)
    points = random_3d_points(n, mean, std)
    ax.plot(points[:,0], points[:,1], points[:,2], 'o')
    plt.show()
