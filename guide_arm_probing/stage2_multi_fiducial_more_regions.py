# Point-pair registration script
# Fiducials are placed on inner, outer and occlusal gingiva surfaces
# Script is used to generate the results reported in Jira task: https://neocis.atlassian.net/browse/HW-9541
# by Xiao Huang @ 07/29/2021

import coordinates
import numpy as np
import region_points
import Readers as Yomiread
import Writers as Yomiwrite
import os
import plot as Yomiplot
import matplotlib.pyplot as plt
import random_points_3d
import registration
import statistics_analysis as sa
import array_processing as ap
import edentulous_arch
import compare


def check_tre(targets, fiducials):
    # define number of trials, mean error and stdev
    tre_total = []
    tre_ang_total = []
    tre_trans_total = []
    x_error = []
    y_error = []
    z_error = []
    original_fiducials = np.copy(fiducials)
    for i in range(N): # for each trial
        # generate noise on fiducials
        fiducials_new = []
        # add biased errors
        if BIAS_ERROR_FLAG == 1:
            center = np.array([-0.154, 0, 0])
            if ORIENTATION == 1:
                fiducials = random_points_3d.biased_3d_points_1(original_fiducials, center, BIAS_MEAN, BIAS_STDEV)
            if ORIENTATION == 2:
                fiducials = random_points_3d.biased_3d_points_2(original_fiducials, center, BIAS_MEAN, BIAS_STDEV)
            if ORIENTATION == 3:
                fiducials = random_points_3d.biased_3d_points_3(original_fiducials, center, BIAS_MEAN, BIAS_STDEV)

        for fiducial in fiducials:
            # select element 0 from the random values
            # add random errors
            # the function random_3d_points only return a matrix, to select a vector, should specify the row number
            fiducial_tem = fiducial + random_points_3d.random_3d_points(1, RAN_MEAN, RAN_STDEV)[0]
            fiducials_new.append(fiducial_tem)

        # perform registration with perturbed fiducials
        R, t = registration.point_set_registration(fiducials_new, original_fiducials)
        # perform registration on targets
        targets_new = []
        if np.ndim(targets) == 1:
            targets_new = np.matmul(R, targets) + t
            tre_tem = np.asarray(targets_new) - targets
            angle_err = np.matmul(R, [0, 0, 1])
            print('angle_err is', angle_err)
            # print('tre vector is', tre)
            tre = np.linalg.norm(tre_tem)
            diff_z = tre_tem[2]
            diff_x = tre_tem[0]
            diff_y = tre_tem[1]
            # print('tre is', tre)
            tre_total.append(tre)
            x_error.append(diff_x)
            y_error.append(diff_y)
            z_error.append(diff_z)

        else:
            tre_sep = []
            tre_ang_sep = []
            tre_trans_sep = []
            x_error_sep = []
            y_error_sep = []
            z_error_sep = []
            for i in range(len(targets)):
                target = targets[i,:]
                target_tem = np.matmul(R, target) + t
                angle_err = compare.comp_rot_misori(R, np.eye(3))
                trans_err = np.linalg.norm(t)
                tre_tem = target_tem - targets[i,:]
                tre = np.linalg.norm(tre_tem)
                diff_z = tre_tem[2]
                diff_x = tre_tem[0]
                diff_y = tre_tem[1]
                x_error_sep.append(diff_x)
                y_error_sep.append(diff_y)
                z_error_sep.append(diff_z)
                tre_sep.append(tre)
                tre_ang_sep.append(angle_err)
                tre_trans_sep.append(trans_err)
            tre_total.append(tre_sep)
            tre_ang_total.append(tre_ang_sep)
            tre_trans_total.append(tre_trans_sep)
            x_error.append(x_error_sep)
            y_error.append(y_error_sep)
            z_error.append(z_error_sep)

    tre_total = np.asarray(tre_total)
    tre_ang_total = np.asarray(tre_ang_total)
    tre_trans_total = np.asarray(tre_trans_total)
    x_error = np.array(x_error)
    y_error = np.array(y_error)
    z_error = np.array(z_error)

    if np.ndim(targets) == 1:
        value_95, mean, std = sa.fit_models_np_plot_mean_std(tre_total)
        print('value 95 is', value_95)
        print('mean is', mean)
        print('std is', std)
    else:
        value_95 = []
        value_95_ang = []
        value_95_trans = []
        value_95_x = []
        value_95_y = []
        value_95_z = []
        for j in range(len(targets)):
            value = sa.fit_models_np_plot(tre_total[:, j])
            value_95.append(value)

            value_ang = sa.fit_models_np_plot(tre_ang_total[:,j])
            value_95_ang.append(value_ang)

            value_trans = sa.fit_models_np_plot(tre_trans_total[:,j])
            value_95_trans.append(value_trans)

            value_x = sa.fit_models_np_plot_mean_std(x_error[:, j])
            value_95_x.append(value_x[0])

            value_y = sa.fit_models_np_plot_mean_std(y_error[:, j])
            value_95_y.append(value_y[0])

            value_z = sa.fit_models_np_plot_mean_std(z_error[:, j])
            value_95_z.append(value_z[0])
        print('value 95 is', value_95)
        print('value 95 angle is', value_95_ang)
        print('value 95 trans is', value_95_trans)
        print('value 95 x is', value_95_x)
        print('value 95 y is', value_95_y)
        print('value 95 z is', value_95_z)
    return value_95, value_95_x, value_95_y, value_95_z


# dynamic grouping
# point is in the format [r, theta, z]
# there are only three regions in radius (inner, occlusal and outer)
#       inner and outer:  z_regions * theta_regions
#       occlusal:   theta_regions
#
# result is region_number which starts from 1
def divide_region(radius_range, angle_range, z_range, point):
    delta_angle = (angle_range[1] - angle_range[0]) / D_ANGLE
    delta_height = (z_range[1] - z_range[0]) / D_HEIGHT
    delta_radius = (radius_range[1] - radius_range[0]) / D_RADIUS
    angle_array = []
    for i in range(D_ANGLE):
        angle_array.append(angle_range[0] + i * delta_angle)
    angle_array.append(angle_range[1])
    radius_array = []
    for i in range(D_RADIUS):
        radius_array.append(radius_range[0] + i * delta_radius)
    radius_array.append(radius_range[1])
    height_array = []
    for i in range(D_HEIGHT):
        height_array.append(z_range[0] + i * delta_height)
    height_array.append(z_range[1])

    # check with region does the point belong to
    r_idx = ap.check_interval_idx_single_value(point[0], radius_array)
    theta_idx = ap.check_interval_idx_single_value(point[1]*180/np.pi, angle_array)
    z_idx = ap.check_interval_idx_single_value(point[2], height_array)

    # due to the definition, r_region is always divided into three sub-regions (inner, occlusal, outer)
    # which gives
    #           r_idx = 0:   inner
    #           r_idx = 1:   outer
    #           r_idx = 2:   occlusal
    #
    # theta_region and height_region are defined differently
    #           theta_idx = 0:  outside
    #           theta_idx > D_ANGLE: outside
    #           theta_idx:   the 'theta_idx'th region in theta regions. I.E., theta_idx = 2, the 2nd theta region

    region_number = -1 # means point is outside of the check region

    if r_idx == 0:  # inner
        if 0 < theta_idx <= D_ANGLE and 0 < z_idx <= D_HEIGHT:
            region_number = (theta_idx-1) * D_HEIGHT + z_idx
    elif r_idx == 1:  # occlusal
        if 0 < theta_idx <= D_ANGLE:
            region_number = D_ANGLE * D_HEIGHT + theta_idx
    else: # outer
        if 0 < theta_idx <= D_ANGLE and 0 < z_idx <= D_HEIGHT:
            region_number = D_ANGLE * D_HEIGHT + D_ANGLE + (theta_idx - 1) * D_HEIGHT + z_idx
    return region_number


if __name__ == "__main__":
    stl_base = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous" \
               "\\Edentulous registration error analysis\\Typodont_scan\\Edentulous_scan\\"
    all_regions = []
    file = stl_base + 'full_arch.txt'
    if os.path.isfile(file):
        print('read txt')
        full_region = Yomiread.read_csv(file,3,-1)
    else:
        stl_file = 'full_arch.stl'
        print('read stl')
        full_region = region_points.read_regions(stl_base + stl_file, voxel_size = 0.05)
        Yomiwrite.write_csv_matrix(stl_base + 'full_arch.txt', full_region)

    base = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous" \
           "\\Edentulous registration error analysis\\Stage2_more_regions\\"
    target_measurement_file = "targets.txt"
    points = Yomiread.read_csv(base + target_measurement_file, 4, 10, flag=1)
    targets_original = points[0:4, 1:4]

    arch1 = edentulous_arch.Edentulous_arch(full_region, targets_original)
    print('angle range is', arch1.target_angle_range)
    print('target position is', arch1.target_origins_cylindrical[:,1]*180/np.pi)
    print('default fiducial is', arch1.default_fiducial)

    # Set up simulation conditions
    D_ANGLE = 1
    D_HEIGHT = 1
    D_RADIUS = 1

    # Select single target or multiple targets simulations (0: single target, 1: multiple targets)
    simulation_flag = 1
    # Define configurations
    configuration = 3
    angle_span = 20
    offset_flag = 0
    location_offset = 15

    # Single Target Simulation
    if simulation_flag == 0:
        arch1.divide_arch(D_RADIUS, D_ANGLE, D_HEIGHT, defined_radius_range=[17, 23], defined_angle_range=[0, 180],
                          defined_height_range=[5, 34], check_for_target=1)
    # Multiple Target Simulation
    if simulation_flag == 1:
        # Configuration I
        if configuration == 1:
            target1_range = [[18.88, 25.27], [18, 26]]
            target2_range = [[17, 23], [14, 24.5]]
            target3_range = [[17, 23], [14, 24.5]]
            target4_range = [[17.46, 23.8], [17, 26]]
            individual_range = [target1_range, target2_range, target3_range, target4_range]
            arch1.divide_arch_individual_target(D_RADIUS, D_ANGLE, D_HEIGHT, defined_angle_range=[0, 180],
                                                individual_defined_range=individual_range)
        # Configuration II
        if configuration == 2:
            target1_range_ma = [[18.88, 25.27], [18, 26], angle_span]
            target2_range_ma = [[17, 23], [14, 24.5], angle_span]
            target3_range_ma = [[17, 23], [14, 24.5], angle_span]
            target4_range_ma = [[17.46, 23.8], [17, 26], angle_span]
            individual_range_modified_angle = [target1_range_ma, target2_range_ma, target3_range_ma, target4_range_ma]

            if offset_flag == 0:
                arch1.divide_arch_individual_target_modified_angle(D_RADIUS, D_ANGLE, D_HEIGHT,
                                                            individual_defined_range=individual_range_modified_angle)
            if offset_flag == 1:
                arch1.divide_arch_individual_target_modified_angle_with_offset(D_RADIUS, D_ANGLE, D_HEIGHT,
                                                            individual_defined_range=individual_range_modified_angle,
                                                            location_offset = 15)
        # Configuration III
        if configuration == 3:
            target1_range_seg = [[24, 29.5], [18, 27], angle_span]
            target2_range_seg = [[17, 23], [14, 24.5], angle_span]
            target3_range_seg = [[17, 23], [14, 24.5], angle_span]
            target4_range_seg = [[17, 23], [14, 24.5], angle_span]
            target5_range_seg = [[22, 29], [17, 26], angle_span]
            individual_range_modified_angle_seg = [target1_range_seg, target2_range_seg, target3_range_seg,
                                               target4_range_seg, target5_range_seg]
            arch1.divide_arch_individual_target_modified_angle_seg(D_RADIUS, D_ANGLE, D_HEIGHT,
                                                individual_defined_range=individual_range_modified_angle_seg,
                                                               angle_span=angle_span)

    # Perform Simulation
    N = 5000
    BIAS_MEAN = 1.0
    BIAS_STDEV = 0.15
    RAN_MEAN = 1.5
    RAN_STDEV = 0.30
    #N_REGION = 10
    THRESHOLD_0 = 0.7
    THRESHOLD_1 = 1
    green_point = []
    yellow_point = []
    red_point = []
    max_value_95 = []
    min_value_95 = []
    total_value_95 = []
    total_value_95_x = []
    total_value_95_y = []
    total_value_95_z = []
    max_x = []
    max_y = []
    max_z = []

    BIAS_ERROR_FLAG = 0
    ORIENTATION = 1
    print('registration')
    # solve TRE
    for i in range(len(arch1.all_regions_cartesion)):
        value_95 = []
        value_95_x = []
        value_95_y = []
        value_95_z = []
        fiducials_new = np.copy(arch1.default_fiducial)
        mm = 0
        for point in arch1.all_regions_cartesion[i]:
            print('checking point ', mm + 1)
            fiducials_new[i,:] = point
            value_95_tem, value_95_x_tem, value_95_y_tem, value_95_z_tem = check_tre(targets_original, fiducials_new)
            value_95.append(value_95_tem)
            value_95_x.append(value_95_x_tem)
            value_95_y.append(value_95_y_tem)
            value_95_z.append(value_95_z_tem)
            if np.ndim(targets_original) == 1:
                if (value_95_tem < THRESHOLD_0):
                    green_point.append(point)
                    print('green shape', np.shape(green_point))
                elif (value_95_tem < THRESHOLD_1):
                    yellow_point.append(point)
                    print('yellow shape', np.shape(yellow_point))
                else:
                    red_point.append(point)
                    print('red shape', np.shape(red_point))
            else:
                if (all(x < THRESHOLD_0 for x in value_95_tem)):
                    green_point.append(point)
                    print('green shape', np.shape(green_point))
                elif (all(x < THRESHOLD_1 for x in value_95_tem)):
                    yellow_point.append(point)
                    print('yellow shape', np.shape(yellow_point))
                else:
                    red_point.append(point)
                    print('red shape', np.shape(red_point))
            mm += 1
        total_value_95.append(value_95)
        total_value_95_x.append(value_95_x)
        total_value_95_y.append(value_95_y)
        total_value_95_z.append(value_95_z)

        max_value_95.append(np.max(value_95, axis=0))
        min_value_95.append(np.min(value_95, axis=0))
        print('maximum 95% value is', max_value_95)
        print('minimum 95% value is', min_value_95)

        max_x.append(np.max(value_95_x, axis=0))
        max_y.append(np.max(value_95_y, axis=0))
        max_z.append(np.max(value_95_z, axis=0))

    # Write to files
    N_REGION = arch1.n_fiducial
    all_regions = arch1.all_regions_cartesion
    for i in range(N_REGION):
        Yomiwrite.write_csv_matrix(base+"Results\\" + np.str(N_REGION) + "_region_" + np.str(RAN_MEAN)
                                   + "_random_tolerance_and " + np.str(BIAS_MEAN) + "_bias_tolerance_value_95_region_"
                                   + np.str(i+1) + ".txt", total_value_95[i])
        Yomiwrite.write_csv_matrix(base + "Results\\" + np.str(N_REGION) + "_region_" + np.str(RAN_MEAN)
                                   + "_random_tolerance_and " + np.str(BIAS_MEAN) + "_bias_tolerance_points_region_"
                                   + np.str(i+1) + ".txt", all_regions[i])
        Yomiwrite.write_csv_matrix(
            base + "Results\\" + np.str(N_REGION) + "_region_" + np.str(RAN_MEAN) + "_random_tolerance_and " + np.str(
                BIAS_MEAN) + "_bias_tolerance_x_error_" + np.str(i + 1) + ".txt", total_value_95_x[i])
        Yomiwrite.write_csv_matrix(
            base + "Results\\" + np.str(N_REGION) + "_region_" + np.str(RAN_MEAN) + "_random_tolerance_and " + np.str(
                BIAS_MEAN) + "_bias_tolerance_y_error_" + np.str(i + 1) + ".txt", total_value_95_y[i])
        Yomiwrite.write_csv_matrix(
            base + "Results\\" + np.str(N_REGION) + "_region_" + np.str(RAN_MEAN) + "_random_tolerance_and " + np.str(
                BIAS_MEAN) + "_bias_tolerance_z_error_" + np.str(i + 1) + ".txt", total_value_95_z[i])

    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    if len(green_point) > 0:
        Yomiplot.plot_3d_points(np.asarray(green_point), ax, color = 'g')
    if len(yellow_point) > 0:
        Yomiplot.plot_3d_points(np.asarray(yellow_point), ax, color = 'y')
    if len(red_point) > 0:
        Yomiplot.plot_3d_points(np.asarray(red_point), ax, color = 'r')
    plt.show()




