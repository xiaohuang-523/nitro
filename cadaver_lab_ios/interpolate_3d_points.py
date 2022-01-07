# Interpolate 3d curve for 3d points
# Use cubic function
# C(x,y) = a + bx + cx2 + dx3 + ey + fy2 + gy3   (Assume d=g)
# Author: Xiao Huang
# Date: 3/23/2021

import numpy as np
import Readers as Yomiread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# calculate cubic function
# input:
#       1. point_3d
#       2. parameter = [a, b, c, d, e, f, g]
def cubic_function(p0, p1, parameter):
    x0 = p0[0]
    y0 = p0[1]
    z0 = p0[2]
    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]
    a = parameter[0]
    b = parameter[1]
    c = parameter[2]
    d = parameter[3]
    e = parameter[4]
    f = parameter[5]
    g = parameter[6]
    C0 = np.array([1, x0, x0**2, x0**3, y0, y0**2, y0**3])
    Cx0 = np.array([])



    C1 = np.array([1, x1, x1**2, x1**3, y1, y1**2, y1**3])
    Cx0 = np.array([0, 1, 2*x0, 3*x0**2, 0, 0, 0])
    Cx1 = np.array([0, 1, 2*x1, ])
    Cx0 = b + 2*c*x0 + 3*d*x0**2
    Cxx0 = 2*c + 6*d*x0
    Cy0 = e + 2*f*y0 + 3*g*y0**2
    Cyy0 = 2*f + 6*g*y0
    C_half = 2*c + 6*d*(x0+x1)/2 - 2*f - 6*g*(y0+y1)/2
    return C0, C1, Cx0, Cxx0, Cy0, Cyy0, C_half


def cubic_matrix(point, point1):
    x = point[0]
    y = point[1]
    z = point[2]
    x1 = point1[0]
    y1 = point1[1]
    z1 = point1[2]
    C = np.array([1, x, x**2, x**3, y, y**2, y**3])
    Cx = np.array([0, 1, 2*x, 3*x**2, 0, 0, 0])
    Cxx = np.array([0, 0, 2, 6*x**2, 0, 0, 0])
    Cy = np.array([[0, 0, 0, 0, 1, 2*y, 3*y**2]])
    Cyy = np.array([0, 0, 0, 0, 0, 2, 6*y**2])
    C_half = np.array([0, 0, 2, 3*(x+x1), 0, -2, -3*(y+y1)])
    matrix = np.array([[1, x, x**2, x**3, y, y**2, y**3],
                       [1, x1, x1 ** 2, x1 ** 3, y1, 1 ** 2, y1 ** 3],
                       [0, 1, 2*x, 3*x**2, 0, 0, 0],
                       [0, 0, 2, 6 * x ** 2, 0, 0, 0],
                       [0, 0, 0, 0, 1, 2 * y, 3 * y ** 2],
                       [0, 0, 0, 0, 0, 2, 6 * y ** 2],
                       [0, 0, 0, 0, 0, 2, 6 * y ** 2],
                       [0, 0, 2, 3 * (x + x1), 0, -2, -3 * (y + y1)]
                       ])
    return matrix


#def interpolate_3d_bezier(pc):


def pp_matlab(breaks, coefficients, number_points):
    n = len(breaks)-1
    pc = np.eye(3)
    for i in range(n):
        print('i is', i)
        t0 = breaks[i]
        t1 = breaks[i+1]
        t = np.linspace(t0, t1, number_points)
        x_coe = coefficients[i * 3,:]
        y_coe = coefficients[i * 3 + 1, :]
        z_coe = coefficients[i * 3 + 2, :]
        print(x_coe)
        print(y_coe)
        print(z_coe)
        x = x_coe[0] * (t-t0)**3 + x_coe[1] * (t-t0)**2 + x_coe[2] * (t-t0) + x_coe[3]
        y = y_coe[0] * (t-t0) ** 3 + y_coe[1] * (t-t0) ** 2 + y_coe[2] * (t-t0) + y_coe[3]
        z = z_coe[0] * (t-t0) ** 3 + z_coe[1] * (t-t0) ** 2 + z_coe[2] * (t-t0) + z_coe[3]
        pc_tem = np.asarray([x, y, z]).transpose()
        pc = np.vstack((pc, pc_tem))
    pc = pc[3:,:]
    return pc.transpose()

# breaks_file = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_rigid_breaks.csv'
# coe_file = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_rigid_coefficients.csv'
# breaks_0 = Yomiread.read_csv(breaks_file, 16)[0]
# coe = Yomiread.read_csv(coe_file, 4)
#
# pc = pp_matlab(breaks_0, coe, 10)
#
#
# fig2 = plt.figure(2)
# ax3d = fig2.add_subplot(111, projection='3d')
# # #ax3d.plot(x_knots, y_knots, z_knots, color = 'blue', label = 'knots')
# ax3d.plot(pc[:, 0], pc[:, 1], pc[:, 2], 'r*')
# #ax3d.plot(x_fine, y_fine, z_fine, color='g', label='fine')
# #ax3d.plot(x_fine_d, y_fine_d, z_fine_d, color='black', label='fine_d')
# fig2.show()
# plt.show()