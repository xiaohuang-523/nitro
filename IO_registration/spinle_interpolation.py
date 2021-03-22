from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def spline_interpolation_3d(spline_p):
    x_sample = spline_p[:, 0]
    y_sample = spline_p[:, 1]
    z_sample = spline_p[:, 2]
    num_true_pts = 30
    tck, u = interpolate.splprep([x_sample, y_sample, z_sample], k=4)
    #x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0, 1, num_true_pts)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    #x_fine_d, y_fine_d, z_fine_d = interpolate.splev(u_fine, tck, der= 1)
    #fine_d = [x_fine_d, y_fine_d, z_fine_d]
    sample = [x_sample, y_sample, z_sample]
    fine = [x_fine, y_fine, z_fine]
    # fig2 = plt.figure(2)
    # ax3d = fig2.add_subplot(111, projection='3d')
    # #ax3d.plot(x_knots, y_knots, z_knots, color = 'blue', label = 'knots')
    # ax3d.plot(x_sample, y_sample, z_sample, 'r*')
    # ax3d.plot(x_fine, y_fine, z_fine, color='g', label='fine')
    # #ax3d.plot(x_fine_d, y_fine_d, z_fine_d, color='black', label='fine_d')
    # fig2.show()
    #
    # fig3 = plt.figure(3)
    # plt.scatter(range(len(x_fine)), x_fine)
    # fig4 = plt.figure(4)
    # plt.scatter(range(len(y_fine)), y_fine)
    # fig5 = plt.figure(5)
    # plt.scatter(range(len(z_fine)), z_fine)
    #
    # fig6 = plt.figure(6)
    # plt.scatter(x_fine, y_fine)
    #
    # fig7 = plt.figure(7)
    # plt.scatter(x_fine, z_fine)
    #
    # plt.legend()
    # plt.show()
    return sample, fine


def spline_interpolation_3d_multiple(spline_list):
    fine_list = []
    sample_list = []
    for spline in spline_list:
        sample_tem, fine_tem = spline_interpolation_3d(spline)
        fine_list.append(fine_tem)
        sample_list.append(sample_tem)
    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')
    # ax3d.plot(x_knots, y_knots, z_knots, 'go')
    ax3d.plot(fine_list[0][0], fine_list[0][1], fine_list[0][2], color ='r', label = 'target spline')
    ax3d.plot(fine_list[1][0], fine_list[1][1], fine_list[1][2], color = 'g', label = 'rigid transformation')
    ax3d.plot(fine_list[2][0], fine_list[2][1], fine_list[2][2], color = 'b', label = 'affine transformation')
    fig2.show()
    fig3 = plt.figure(3)
    ax3d = fig3.add_subplot(111, projection='3d')
    ax3d.plot(sample_list[0][0], sample_list[0][1], sample_list[0][2], 'r*', label = 'target spline')
    ax3d.plot(sample_list[1][0], sample_list[1][1], sample_list[1][2], 'g*', label = 'rigid transformation')
    ax3d.plot(sample_list[2][0], sample_list[2][1], sample_list[2][2], 'b*', label = 'affine transformation')
    fig3.show()
