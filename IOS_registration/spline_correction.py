# This script is used to perform spline grid matching
# Divide the spline into pieces and apply only the translation to match the splines.
# by Xiao Huang, 03/22/2021

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Readers as Yomiread
from scipy import interpolate
import coordinates


def dis(p1, p2):
    delta = np.asarray(p2 - p1)
    distance = np.linalg.norm(delta)
    return distance

def arc_length(x, y):
    arc = np.sqrt((x[1] - x[0])**2 + (y[1]-y[0])**2)
    for i in range(len(x)-2):
        arc = arc + np.sqrt((x[i+2] - x[i+1])**2 + (y[i+2]-y[i+1])**2)
    return arc

def arc_segment(x, y, arc_total_length, segment_number):
    arc = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
    count = 1
    segment_points_x = []
    segment_points_y = []
    for i in range(len(x) - 2):
        arc = arc + np.sqrt((x[i + 2] - x[i + 1]) ** 2 + (y[i + 2] - y[i + 1]) ** 2)
        if arc > count * arc_total_length / segment_number:
            count += 1
            segment_points_x.append(x[i])
            segment_points_y.append(y[i])
    segment_points = np.asarray([segment_points_x, segment_points_y]).transpose()

    return segment_points


def combine_pc(list):
    combined_pc = list[0]
    for i in range(len(list)-1):
        #combined_pc = np.vstack((combined_pc, list[i+1]))
        combined_pc = np.concatenate((combined_pc,list[i+1]))
    return combined_pc

# Check which interval index a value is
# References: https://stackoverflow.com/questions/34798343/fastest-way-to-check-which-interval-index-a-value-is
# Only works for ascending order
def check_interval_idx(values, intervals):
    idx = []
    for value in values:
        idx.append(intervals.size - np.searchsorted(intervals[::-1], value, side="right")) # sort descending order
    return idx


def check_interval_idx_single_value(value, intervals):
    #print('interval is', intervals)
    if intervals[0] < intervals[-1]:
        idx = np.searchsorted(intervals, value, side="left")  # ascending order
    if intervals[0] > intervals[-1]:
        idx = intervals.size - np.searchsorted(intervals[::-1], value, side="right")   # descending order
    return idx


# Convert to cylindrical coordinates
def check_curvilinear_boundary(spline, normal_axis = 2):
    center_x = (spline[0][0] + spline[0][-1]) / 2
    center_y = (spline[1][0] + spline[1][-1]) / 2
    theta = np.zeros(len(spline[0]))
    r = np.zeros(len(spline[0]))
    for i in range(len(spline[0])):
        dy = spline[1][i] - center_y
        dx = spline[0][i] - center_x
        theta[i] = np.arctan2(dy, dx)
        if theta[i] < 0:
            theta[i] = theta[i] + 2*np.pi
        r[i] = np.sqrt(dx**2 + dy**2)
    center = [center_x, center_y]
    boundary = [r, theta]
    return center, boundary


# Convert to cylindrical coordinates
def convert_cylindrical_2D(pc, center):
    theta = []
    r = []
    for point in pc:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        angle = np.arctan2(dy, dx)
        #if angle < 0:
        #    angle = angle + 2*np.pi
        theta.append(angle)
        r.append(np.sqrt(dx**2 + dy**2))
    pc_cylindrical = np.asarray([r, theta])
    return pc_cylindrical.transpose()


# Perform point displacement
def displacement(pc_to_move, pc_to_move_cylindrical, spline_boundary, displacement):
    pc_moved = []
    pc_to_move_ = np.copy(pc_to_move)
    for i in range(pc_to_move.shape[0]):
        point = pc_to_move_[i,:]
        theta = pc_to_move_cylindrical[i, 1]
        idx = check_interval_idx_single_value(theta, spline_boundary)
        if idx > np.shape(displacement)[0]-1:
            move = np.array([displacement[-1, 0], displacement[-1, 1], displacement[-1, 2]])
        else:
            move = np.array([displacement[idx, 0], displacement[idx, 1], displacement[idx, 2]])
        new_point = point + move
        pc_moved.append(new_point)
        #print('point is', point)
        #print('new point is', new_point)
        #print('move is', move)
        del move
        del new_point
    return np.asarray(pc_moved)


def displacement_single_point(pc_to_move, pc_to_move_cylindrical, spline_boundary, displacement):
    pc_moved = []
    pc_to_move_ = np.copy(pc_to_move)

    point = pc_to_move_
    theta = pc_to_move_cylindrical[1]
    idx = check_interval_idx_single_value(theta, spline_boundary)
    if idx > np.shape(displacement)[0]-1:
        move = np.array([displacement[-1, 0], displacement[-1, 1], displacement[-1, 2]])
        #move = np.array([displacement[-1, 0], displacement[-1, 1], 0])
    else:
        move = np.array([displacement[idx, 0], displacement[idx, 1], displacement[idx, 2]])
        #move = np.array([displacement[idx, 0], displacement[idx, 1], 0])
    new_point = point + move
    pc_moved = new_point
    return np.asarray(pc_moved)



# Perform point displacement
def displacement_partial(pc_to_move, pc_to_move_cylindrical, spline_boundary, displacement):
    pc_moved = []
    pc_to_move_ = np.copy(pc_to_move)
    for i in range(pc_to_move.shape[0]):
        point = pc_to_move_[i,:]
        theta = pc_to_move_cylindrical[i, 1]
        idx = check_interval_idx_single_value(theta, spline_boundary)
        if idx > np.shape(displacement)[0]-1:
            move = np.array([0, 0, 0])
        elif idx == 0:
            move = np.array([0, 0, 0])
        else:
            move = np.array([displacement[idx, 0], displacement[idx, 1], displacement[idx, 2]])
        new_point = point + move
        pc_moved.append(new_point)
        #print('point is', point)
        #print('new point is', new_point)
        #print('move is', move)
        del move
        del new_point
    return np.asarray(pc_moved)


# Perform point displacement
def displacement_partial_version2(pc_to_move, pc_to_move_cylindrical, spline_boundary, ignore_boundary, displacement):
    pc_moved = []
    pc_to_move_ = np.copy(pc_to_move)
    for i in range(pc_to_move.shape[0]):
        point = pc_to_move_[i,:]
        theta = pc_to_move_cylindrical[i, 1]
        if theta > ignore_boundary[0] and theta < ignore_boundary[1]:
            move = np.array([0,0,0])
        else:
            idx = check_interval_idx_single_value(theta, spline_boundary)
            if idx > np.shape(displacement)[0]-1:
                move = np.array([displacement[-1, 0], displacement[-1, 1], displacement[-1, 2]])
            elif idx == 0:
                move = np.array([displacement[0, 0], displacement[0, 1], displacement[0, 2]])
            else:
                move = np.array([displacement[idx, 0], displacement[idx, 1], displacement[idx, 2]])
        new_point = point + move
        pc_moved.append(new_point)
        del move
        del new_point
    return np.asarray(pc_moved)



# Split curve into equal pieces
# Work with fitted spline curve using scipy spline curve fitting function.
# Return a list of points which segment the curve into equal pieces, works better for more segmenting points
def fit_spline_and_split(spline_points, spline_points_cylindrical_center, n_guide_points):
    x = spline_points[:,0]
    y = spline_points[:,1]
    z = spline_points[:,2]
    f_spline_yx = interpolate.interp1d(x, y, kind='cubic', fill_value="extrapolate")
    f_spline_zx = interpolate.interp1d(x, z, kind='cubic', fill_value="extrapolate")
    # offset_end_points = 0.5
    grid_size = 0.01
    xnew = np.arange(np.min(x), np.max(x), grid_size)
    ynew = f_spline_yx(xnew)
    znew = f_spline_zx(xnew)

    grid_points = np.asarray([xnew, ynew]).transpose()
    grid_points_cylindrical = convert_cylindrical_2D(grid_points, spline_points_cylindrical_center)
    guide_points_theta = np.arange(0, np.pi, np.pi/n_guide_points)
    guide_points_idx = check_interval_idx(guide_points_theta, grid_points_cylindrical[:,1])
    guide_points_xy = []
    guide_points_z = []
    for idx in guide_points_idx:
        if idx < grid_points.shape[0]:
            guide_points_xy.append(grid_points[idx,:])
            guide_points_z.append(znew[idx])
    guide_points_xy = np.asarray(guide_points_xy)
    guide_points_z = np.reshape(guide_points_z, (len(guide_points_z), 1))
    print('guide_points z shape is', np.shape(guide_points_z))
    guide_points = np.hstack((guide_points_xy, guide_points_z))
    return guide_points, guide_points_theta


# Split curve into equal pieces
# Work with fitted spline curve using scipy spline curve fitting function.
# Return a list of points which segment the curve into equal pieces, works better for more segmenting points
def fit_spline_and_split_2(spline_points, segment_number = 50):
    x = spline_points[:,0]
    y = spline_points[:,1]
    z = spline_points[:,2]

    tck_xy = interpolate.splrep(x, y, s=0)
    tck_xz = interpolate.splrep(x, z, s=0)
    deltax = 1e-5
    spline_fine_point = []
    for j in range(len(x)-1):
        start = x[j]
        end = x[j+1]
        segment_points = np.zeros((segment_number,3))
        x_tem = np.arange(start, end, deltax)
        y_tem = interpolate.splev(x_tem, tck_xy, der=0)
        z_tem = interpolate.splev(x_tem, tck_xz, der=0)
        length_xy = arc_length(x_tem, y_tem)
        length_xz = arc_length(x_tem, z_tem)

        segments_points_xy = arc_segment(x_tem, y_tem, length_xy, 50)
        segments_points_xz = arc_segment(x_tem, z_tem, length_xz, 50)

        # the fine parts start from the first spline control points + 49 segment points
        segment_points[0, 0] = x[j]
        segment_points[0, 1] = y[j]
        segment_points[0, 2] = z[j]
        for i in range(segment_number-1):
            segment_points[i+1, 0] = segments_points_xy[i, 0]
            segment_points[i+1, 1] = segments_points_xy[i, 1]
            segment_points[i+1, 2] = segments_points_xz[i, 1]

        if j == len(x) -2:
            segment_points = np.vstack((segment_points, np.array([x[j+1], y[j+1], z[j+1]])))
        spline_fine_point.append(segment_points)
        print('shape of spline_fine_point is', np.shape(spline_fine_point))

    return combine_pc(spline_fine_point)



# test_pc = Yomiread.read_csv("G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_target_points.csv", 3, -1)
# sample, fine = sp.spline_interpolation_3d(test_pc)
#
# # find occlusal normal direction
# delta = []
# for i in range(3):
#     delta.append(np.abs(np.max(fine[i]) - np.min(fine[i])))
# normal_idx = np.where(np.asarray(delta) < 10)[0][0]
#
# # Assume the full arch is placed in normal orientation
# center_x = (fine[0][0] + fine[0][-1]) / 2
# center_y = (fine[1][0] + fine[1][-1]) / 2
#
# boundary = np.zeros(len(fine[0]))
# for i in range(len(fine[0])):
#     dy = fine[1][i] - center_y
#     dx = fine[0][i] - center_x
#     cos = dy / np.sqrt(dx**2 + dy**2)
#     boundary[i] = cos
#
# test = np.linspace(0,1.,8)
#
# print('test is', test)
# print('boundary is', boundary)
# idx = check_interval_idx(test, boundary)
# print('idx is', idx)
#
#
#
# fig = plt.figure()
# plt.scatter(range(len(boundary)), boundary, label = 'grid')
#
# fig2 = plt.figure()
# plt.scatter(fine[0], fine[1], label = 'fine')
# plt.legend()
# plt.show()
# exit()
#
# #yder = interpolate.splev(xnew, tck, der=1)
#
# fig1 = plt.figure()
# plt.scatter(sample[1,:], sample[2,:], color='g', label='sample grid')
# plt.scatter(fine[1,:], fine[2,:], color='r', label='fine grid')
# plt.legend()
# plt.show()
