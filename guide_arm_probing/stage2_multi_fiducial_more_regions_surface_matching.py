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
import ICP
import local_registration as local_ICP


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

    # Define configurations
    angle_span = 20

    # Prepare surface matching
    arch1.divide_arch_individual_target_surface_matching(angle_span)
    print('shape of arch1 surface points is', np.shape(arch1.surface_points))
    print('shape of arch1 surface points is', arch1.surface_points)
    original_surface_points = arch1.surface_points

    # Perform Simulation
    N = 400
    BIAS_MEAN = 1.0
    BIAS_STDEV = 0.15
    RAN_MEAN = 1.5
    RAN_STDEV = 0.30
    TRANS_INIT = np.eye(4)
    VOXEL_SIZE_ICP = 0.05
    THRESHOLD_ICP = 50
    RMS_LOCAL_REGISTRATION = 0.2


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
    tre_total = []
    tre_ang_total = []
    tre_trans_total = []
    x_error = []
    y_error = []
    z_error = []


    for i in range(N):
        # add random noise to all points
        print('iteration ', i+1)
        turbulent_surface_points = []
        rms = []
        for point in original_surface_points:
            # the function random_3d_points only return a matrix, to select a vector, should specify the row number
            point_tem = point + random_points_3d.random_3d_points(1, RAN_MEAN, RAN_STDEV)[0]
            turbulent_surface_points.append(point_tem)
        turbulent_surface_points = np.asarray(turbulent_surface_points)
        # perform ICP surface matching
        ICP_single = local_ICP.registration(VOXEL_SIZE_ICP, THRESHOLD_ICP, original_surface_points, turbulent_surface_points, TRANS_INIT)
        rms.append(ICP_single.inlier_rmse)
        transformation = ICP_single.transformation

        # check TRE
        tre_sep = []
        tre_ang_sep = []
        tre_trans_sep = []
        x_error_sep = []
        y_error_sep = []
        z_error_sep = []
        for i in range(len(arch1.target_origins_cartesian)):
            target = arch1.target_origins_cartesian[i, :]
            target_homo = np.insert(target, 3, 1)
            target_tem = np.matmul(transformation, target_homo)[0:3]
            angle_err = compare.comp_rot_misori(transformation[0:3,0:3], np.eye(3))
            trans_err = np.linalg.norm(transformation[0:3,3])
            tre_tem = target_tem - arch1.target_origins_cartesian[i, :]
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
    tre_trans_total = np.asarray(tre_trans_total)
    tre_ang_total = np.asarray(tre_ang_total)
    x_error = np.asarray(x_error)
    y_error = np.asarray(y_error)
    z_error = np.asarray(z_error)

    value_95 = []
    value_95_ang = []
    value_95_trans = []
    value_95_x = []
    value_95_y = []
    value_95_z = []
    for j in range(len(arch1.target_origins_cartesian)):
        value = sa.fit_models_np_plot(tre_total[:, j])
        value_95.append(value)

        value_ang = sa.fit_models_np_plot(tre_ang_total[:, j])
        value_95_ang.append(value_ang)

        value_trans = sa.fit_models_np_plot(tre_trans_total[:, j])
        value_95_trans.append(value_trans)

        #value_x = sa.fit_models_np_plot_mean_std(x_error[:, j])
        #value_95_x.append(value_x[0])

        #value_y = sa.fit_models_np_plot_mean_std(y_error[:, j])
        #value_95_y.append(value_y[0])

        #value_z = sa.fit_models_np_plot_mean_std(z_error[:, j])
        #value_95_z.append(value_z[0])
    print('value 95 is', value_95)
    print('value 95 angle is', value_95_ang)
    print('value 95 trans is', value_95_trans)
    print('value 95 x is', value_95_x)
    print('value 95 y is', value_95_y)
    print('value 95 z is', value_95_z)



    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    Yomiplot.plot_3d_points(original_surface_points, ax, color = 'g')
    Yomiplot.plot_3d_points(turbulent_surface_points, ax, color = 'blue')

    plt.show()




