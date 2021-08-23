# Information regarding multi-normal distribution in python
# source1:  https://docs.scipy.org/doc//numpy-1.10.4/reference/generated/numpy.random.multivariate_normal.html
# source2:  https://stackoverflow.com/questions/25720600/generating-3d-gaussian-distribution-in-python
# source3:
# https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
# source4(good):
# https://stackoverflow.com/questions/22439193/how-to-generate-new-points-as-offset-with-gaussian-distribution-for-some-points

import numpy as np
import Readers as Yomiread
import random_points_3d
import registration
import region_points
import Writers as Yomiwrite

from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import weibull_min
from scipy.stats import beta
from scipy.stats import burr
import scipy.stats as stats
import matplotlib.pyplot as plt

import statistics_analysis as sa

import plot as Yomiplot
import os


def check_tre(targets, fiducials):
    # define number of trials, mean error and stdev
    tre_total = []
    tre_1 = []
    tre_2 = []
    tre_3 = []
    tre_4 = []
    original_fiducials = np.copy(fiducials)
    #print('original_fiducials are', original_fiducials)
    for i in range(N): # for each trial
        # generate noise on fiducials
        fiducials_new = []
        # add biased errors
        #print('loop', i)
        if BIAS_ERROR_FLAG == 1:
            center = np.array([-0.154, 0, 0])
            if ORIENTATION == 1:
                #print('fiducials before biased errors are', original_fiducials)
                fiducials = random_points_3d.biased_3d_points_1(original_fiducials, center, BIAS_MEAN, BIAS_STDEV)
                #print('fiducials after biased errors are', fiducials)
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
        #print('fiducials_after random errors are', fiducials_new)
        #print('fiducials original check are', original_fiducials)
        #print('fiducial error vector is', np.linalg.norm(fiducials_new - original_fiducials, axis=1))
        R, t = registration.point_set_registration(fiducials_new, original_fiducials)
        #print('R is', R)
        #print('t is', np.linalg.norm(t))
        # perform registration on targets
        targets_new = []
        if np.ndim(targets) == 1:
            targets_new = np.matmul(R, targets) + t
            tre = np.asarray(targets_new) - targets
            # print('tre vector is', tre)
            tre = np.linalg.norm(tre)
            # print('tre is', tre)
            tre_total = np.append(tre_total, tre)
            tre_sep = tre_total
            #value_95 = sa.fit_models_np_plot(tre_sep)


        else:
            for target in targets:
                target_tem = np.matmul(R, target) + t
                targets_new.append(target_tem)

                tre = np.asarray(targets_new) - targets
        #print('tre vector is', tre)
                tre = np.linalg.norm(tre, axis=1)
        #print('tre is', tre)
                tre_total = np.append(tre_total, tre)
                tre_1.append(tre[0])
                tre_2.append(tre[1])
                tre_3.append(tre[2])
                tre_4.append(tre[3])


    if np.ndim(targets) == 1:
        tre_sep = tre_total
        value_95 = sa.fit_models_np_plot(tre_sep)
        print('value 95 is', value_95)
    else:
        value_95 = []
        tre_sep = [tre_1, tre_2, tre_3, tre_4]
        for j in range(len(targets)):
            value = sa.fit_models_np_plot(tre_sep[j])
            value_95.append(value)
    return value_95


if __name__ == '__main__':

    # read file for true fiducials and targets
    base = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\Edentulous registration error analysis\\Stage2\\"
    target_measurement_file = "1_target_5_fiducials.txt"
    points = Yomiread.read_csv(base + target_measurement_file, 4, 10, flag=1)
    targets_original = points[0, 1:4] + [0, 0, 4]
    fiducials_original = points[1:6, 1:4]

    print('targets_original', targets_original)


    N = 5000
    BIAS_MEAN = 1.0
    BIAS_STDEV = 0.15
    RAN_MEAN = 1.2
    RAN_STDEV = 0.15
    N_REGION = 5
    THRESHOLD_0 = 0.7
    THRESHOLD_1 = 1
    green_point = []
    yellow_point = []
    red_point = []
    max_value_95 = []
    min_value_95 = []
    total_value_95 = []

    BIAS_ERROR_FLAG = 0
    ORIENTATION = 1

    stl_base = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\Edentulous registration error analysis\\Typodont_scan\\Edentulous_scan\\"
    all_regions = []

    for i in range(5):
        file = stl_base + 'region_points_' + np.str(i+1) + '.txt'
        if os.path.isfile(file):
            print('read txt')
            individual_region = Yomiread.read_csv(file,3,-1)
        else:
            stl_file = 'Fiducial_region_' + np.str(i+1) + '.stl'
            print('read stl')
            individual_region = region_points.read_regions(stl_base + stl_file, voxel_size = 2.0)
            Yomiwrite.write_csv_matrix(stl_base + 'region_points_' + np.str(i + 1) + '.txt', individual_region)
        all_regions.append(individual_region)

    for i in range(len(all_regions)):
        value_95 = []
        fiducials_new = np.copy(fiducials_original)
        mm = 0
        for point in all_regions[i]:
            print('checking point ', mm + 1)
            fiducials_new[i,:] = point
            #print('fiducials are ', fiducials_new)
            value_95_tem = check_tre(targets_original, fiducials_new)
            value_95.append(value_95_tem)
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
        max_value_95.append(np.max(value_95))
        min_value_95.append(np.min(value_95))
        print('maximum 95% value is', max_value_95)
        print('minimum 95% value is', min_value_95)
    #value_95, model = check_tre(targets_original, fiducials_original)

    #green_file = "Results\\" + np.str(N_REGION) + "_region_" + np.str(RAN_MEAN) + "_random_tolerance_green.txt"
    #yellow_file = "Results\\" + np.str(N_REGION) + "_region_" + np.str(MEAN) + "_tolerance_yellow.txt"
    #red_file = "Results\\" + np.str(N_REGION) + "_region_" + np.str(MEAN) + "_tolerance_red.txt"

    #Yomiwrite.write_csv_matrix(base+green_file, green_point)
    #Yomiwrite.write_csv_matrix(base + yellow_file, yellow_point)
    #Yomiwrite.write_csv_matrix(base + red_file, red_point)

    for i in range(N_REGION):
        Yomiwrite.write_csv_matrix(base+"Results\\" + np.str(N_REGION) + "_region_" + np.str(RAN_MEAN) + "_random_tolerance_and " + np.str(BIAS_MEAN) + "_bias_tolerance_value_95_region_" + np.str(i+1) + ".txt", total_value_95[i])
        Yomiwrite.write_csv_matrix(base + "Results\\" + np.str(N_REGION) + "_region_" + np.str(RAN_MEAN) + "_random_tolerance_and " + np.str(BIAS_MEAN) + "_bias_tolerance_points_region_" + np.str(i+1) + ".txt", all_regions[i])

    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    if len(green_point) > 0:
        Yomiplot.plot_3d_points(np.asarray(green_point), ax, color = 'g')
    if len(yellow_point) > 0:
        Yomiplot.plot_3d_points(np.asarray(yellow_point), ax, color = 'y')
    if len(red_point) > 0:
        Yomiplot.plot_3d_points(np.asarray(red_point), ax, color = 'r')
    # Yomiplot.plot_3d_points(all_vertex[3], ax, color = 'g')
    plt.show()


    exit()


    # define number of trials, mean error and stdev
    n = 10000
    mean = 0
    stdev = 0.15
    fiducial_pc = []
    target_pc = []
    tre_total = []
    tre_1 = []
    tre_2 = []
    tre_3 = []
    tre_4 = []
    for i in range(n): # for each trial
        # generate noise on fiducials
        fiducials_new = []
        for fiducial in fiducials_original:
            # select element 0 from the random values
            # the function random_3d_points only return a matrix, to select a vector, should specify the row number
            fiducial_tem = fiducial + random_points_3d.random_3d_points(1, mean, stdev)[0]
            fiducials_new.append(fiducial_tem)

        # perform registration with perturbed fiducials
        R, t = registration.point_set_registration(fiducials_new, fiducials_original)

        # perform registration on targets
        targets_new = []
        for target in targets_original:
            target_tem = np.matmul(R, target) + t
            targets_new.append(target_tem)

        tre = np.asarray(targets_new) - targets_original
        tre = np.linalg.norm(tre, axis=1)
        tre_total = np.append(tre_total, tre)
        tre_1.append(tre[0])
        tre_2.append(tre[1])
        tre_3.append(tre[2])
        tre_4.append(tre[3])

    tre_sep = [tre_1, tre_2, tre_3, tre_4]
    #print('tre_sep is', tre_sep)

    value_95 = []
    models = []
    for j in range(len(targets_original)):
        value, model = sa.fit_models(tre_sep[j])
        #print('95% value is', value)
        #print('model is', model)
        value_95.append(value)
        models.append(model)

    print('value_95', value_95)
    print('models are', models)






    # fig1 = plt.figure()
    # y, bins, p = plt.hist(tre_total, bins=20, density=True, alpha=0.6)
    # fig2 = plt.figure()
    # y, bins, p = plt.hist(tre_1, bins=20, density=True, alpha=0.6)
    # fig3 = plt.figure()
    # y, bins, p = plt.hist(tre_2, bins=20, density=True, alpha=0.6)
    # fig4 = plt.figure()
    # y, bins, p = plt.hist(tre_3, bins=20, density=True, alpha=0.6)
    # fig5 = plt.figure()
    # y, bins, p = plt.hist(tre_4, bins=20, density=True, alpha=0.6)
    # plt.show()



