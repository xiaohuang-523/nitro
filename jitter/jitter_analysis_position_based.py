import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy import signal
import os
import functools
import re


# Read joint angles from the first Jog to the last Jog.
# The length of the vector is set as 36800 based on the standard method requirements. (For 3 deg/s tests)
def getJointAngle_separate(logfile,idx1, idx2):
    # Check the format of log files
    qra = []
    qrd = []
    qra_line = []
    # Read data
    for i in range(5):
        data_tem = readLog_JointAngle(logfile, idx1[i+1], idx2[i+2])
        qra_tem = np.asarray(data_tem[0])
        qrd_tem = np.asarray(data_tem[1])
        qra_line_tem = np.asarray(data_tem[2])
        qra.append(qra_tem)
        qrd.append(qrd_tem)
        qra_line.append(qra_line_tem)
    return qra, qrd, qra_line


def getJointAngle_entire(logfile,idx1, idx2):
    # Check the format of log files
    # Read data
    data_tem = readLog_JointAngle(logfile, idx1[1], idx2[6])
    qra = np.asarray(data_tem[0])
    qrd = np.asarray(data_tem[1])
    return qra, qrd



# Read log file
def readLog_JointAngle(log_file_path, startLine, endLine):
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    QRA_list = []
    QRD_list = []
    QRA = []
    QRD = []
    QRA_line = []
    QRD_line = []

    for line in range(startLine,endLine):
        if "QRA" in FILE[line]:
            QRA_list.append(FILE[line][65:])
            QRA_line.append(line)
        if "QRD" in FILE[line]:
            QRD_list.append(FILE[line][65:])
            QRD_line.append(line)

    for j in range(len(QRA_list)):
        QRA_data_t = np.fromstring(QRA_list[j], dtype=float, sep=',')
        QRA.append(QRA_data_t)

    for k in range(len(QRD_list)):
        QRD_data_t = np.fromstring(QRD_list[k], dtype=float, sep=',')
        QRD.append(QRD_data_t)

    return np.asarray(QRA), np.asarray(QRD), np.asarray(QRA_line), np.asarray(QRD_line)


# check the idx of the line which contains 'str'
# The output is an array with all idx
# idx[2] is the paused status at pose 2
def check_idx_log_api_mode_pause(log_file_path, str="API_CM_MODE switched to 576"):
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    idx_end = len(FILE) -1
    idx = []
    for line in range(len(FILE)):
        if str in FILE[line]:
            idx_tem = line
            idx.append(idx_tem)
    #idx.append(idx_end)
    idx = np.asarray(idx)
    return idx


# check the idx of the line which contains 'str'
# The output is an array with all idx
# idx[1] is the start jogging to pose 2.
def check_idx_log_api_mode_start(log_file_path, str="API_CM_MODE switched to 592"):
    # str = "Set Jog To mode"
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    idx_end = len(FILE) -1
    idx = []
    for line in range(len(FILE)):
        if str in FILE[line]:
            idx_tem = line
            idx.append(idx_tem)
    #idx.append(idx_end)
    idx = np.asarray(idx)
    return idx


# Check whether the log files are in the standard format
# Standard format 6 jogs to 6 different poses
# Poses are
#    0.58793,0.27922,-1.66853,-0.7877,-1.64784,0.72724
#    0.49545,1.79998, -0.21207,-1.3465,-1.81774,0.47881
#    -0.58791,0.29999,-1.66857,0.78777,-1.64785,-0.72997
#    -0.55587,1.80001,-0.33414,1.40462,-2.14658,-0.89494
#    0.5879,0.29997,-1.66855,-0.78767,-1.64789,0.72724
#    0.55336,1.79999,-0.33219,-1.5266,0.11643,0.90617
def check_logfiles(log_file_path, idx):
    # Define target pose list
    target_list = np.array([[0.58793,0.27922,-1.66853,-0.7877,-1.64784,0.72724],
                            [0.49545,1.79998, -0.21207,-1.3465,-1.81774,0.47881],
                            [-0.58791,0.29999,-1.66857,0.78777,-1.64785,-0.72997],
                            [-0.55587,1.80001,-0.33414,1.40462,-2.14658,-0.89494],
                            [0.5879,0.29997,-1.66855,-0.78767,-1.64789,0.72724],
                            [0.55336,1.79999,-0.33219,-1.5266,0.11643,0.90617]])

    # read log file
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()

    print('checking the format of log files')
    print('length of idx is', len(idx))
    # check all 6 jog motions
    if len(idx) < 7:
        print('Error, not enough jog motions are found.\n '
              'Please check the log files and make sure only the neolib_vs.log files are presented\n'
              'The execution is terminated')
        exit()

    # check the target position of each jogging
    count = 0
    for line in idx[:-1]:
        count += 1
        target_tem = FILE[line][107:]

        # remove the parathesis from the list
        target_tem = re.sub(r'[()]', '', target_tem)
        target_tem_value = np.fromstring(target_tem, dtype=float, sep=',')
        # compare whether the two lists are identical.
        if functools.reduce(lambda i, j : i and j, map(lambda m, k: m == k, target_tem_value, target_list[count-1,:]), False):
            print('Jog motion ' + np.str(count) + ' is with wrong target position')
            print('Please check the log files and make sure only the neolib_vs.log files are presented.\n'
                  'The execution is terminated')
            exit()
    print('The log file is in the correct format')


# Analyze jittering using the standard method and log files
def analyze_jittering_single_total_log(logfile, str, cutoff, fs, order):
    # Check index based on the string
    idx1 = check_idx_log_api_mode_start(logfile)
    idx2 = check_idx_log_api_mode_pause(logfile)
    print('idx 592 is', idx1)
    print('idx 576 is', idx2)

    # Define # of joints
    joint_n = 6

    # Initialize the stdev variable
    std_1 = np.zeros(6)

    # Set filter parameters
    cutoff = cutoff
    fs = fs
    order = order
    print('Reading values')
    # Read joint angles from log files (Using standard method)
    qra, qrd, qra_line = getJointAngle_separate(logfile,idx1, idx2)

    # for o in range(5):
    #     print('qra 100 angels are ', qra[o][100,:])
    #     print('qra line 100 angles line number are', qra_line[o][100])
    #     print('qra last is', qra[o][-1:,:])
    #     print('qra last line number are', qra_line[o][-1])
    #     print('# of joints are ', np.shape(qra[o]))

    # Check if qra1 and qrd1 are of the same length
    if len(qra) != len(qrd):
        print('Error, lengths of QRA and QRD are different')
        exit()
    fluct = []
    #fluct_total = np.zeros(6)
    for i in range(len(qra)):
        qra_tem = qra[i]
        qrd_tem = qrd[i]
        # Solve the bugs where two variables have different lengths.
        if np.shape(qra_tem)[0] == np.shape(qrd_tem)[0]:
            fluctuation_tem = qra_tem - qrd_tem
        elif np.shape(qra_tem)[0] > np.shape(qrd_tem)[0]:
            qra_tem = qra_tem[:-1, :]
            fluctuation_tem = qra_tem - qrd_tem
        elif np.shape(qra_tem)[0] < np.shape(qrd_tem)[0]:
            qrd_tem = qrd_tem[:-1, :]
            fluctuation_tem = qra_tem - qrd_tem
        fluct.append(fluctuation_tem)
        #fluct_total = np.vstack((fluct_total, fluct_tem))

    qra2, qrd2 = getJointAngle_entire(logfile, idx1, idx2)
    # Solve the bugs where two variables have different lengths.
    if np.shape(qra2)[0] == np.shape(qrd2)[0]:
        fluct_total = qra2 - qrd2
    elif np.shape(qra2)[0] > np.shape(qrd2)[0]:
        qra2 = qra2[:-1, :]
        fluct_total = qra2 - qrd2
    elif np.shape(qra2)[0] < np.shape(qrd2)[0]:
        qrd2 = qrd2[:-1, :]
        fluct_total = qra2 - qrd2



    for m in range(5):
        print('The 100th is of jog ' + np.str(m+1)+' is', fluct[m][100,:])
        #print('last 10 joint angles are', fluct[m][-10:,:])
    #    print('# of data in jog ' +np.str(m+1)+' is' , np.shape(fluct[m]))

    #fluct_total = qra2 - qrd2
    print('size of fluct_total is', np.shape(fluct_total))
    print('')

    # For each joint, apply high pass filter
    for i in range(joint_n):
        fluct_filtered = butter_highpass_filter(np.asarray(fluct_total)[:,i], cutoff, fs, order)
        std1_tem = np.std(fluct_filtered)
        std_1[i] = std1_tem

    # For each path, apply high pass filter to eah path
    std_sep = np.zeros((5,6))
    for path in range(5):
        for joint in range(joint_n):
            #print('shape of fluct is', np.shape(np.asarray(fluct[path])))
            fluct_sep_filtered = butter_highpass_filter(np.asarray(fluct[path])[:,joint], cutoff, fs, order)
            std_sep_tem = np.std(fluct_sep_filtered)
            std_sep[path,joint] = std_sep_tem
    #print('std_sep is', std_sep)

    return std_1, std_sep, qra, qrd


def butter_highpass_filter(data, cutoff, fs, order=6):
    ba = butter_highpass(cutoff, fs, order=order)
    filtered = signal.lfilter(ba[0], ba[1], data)
    return filtered


def butter_highpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    ba = signal.butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return ba


# main program
def main():
    std_allfile = []
    std_sep_allfile = []
    count = 0

    file_result_path = file_path +'/results/'
    if not os.path.exists(file_result_path):
        os.makedirs(file_result_path)

    for file in os.listdir(file_path):
        print('')
        print('Loading file', file_path +'/'+ file)
        # use log files
        if file.endswith('.log'):
            count += 1
            file_name = file_path +'/'+ file
            # ------Using standard method with log file------ #
            std_tem, std_sep, qra_tem, qrd_tem = analyze_jittering_single_total_log(file_name, str, cutoff, fs, order)
            std_allfile.append(std_tem)
            std_sep_allfile.append(std_sep)

    std_allfile = np.asarray(std_allfile)
    std_sep_allfile = np.asarray(std_sep_allfile)
    #print('std_sep_allfile is', std_sep_allfile)

    # Old Thresholds [1.36E-05,	8.94E-05, 7.33E-05,	3.54E-05, 1.98E-05,	1.48E-05]
    # New Thresholds [1.39E-05,	8.38E-05, 7.71E-05,	3.42E-05, 2.14E-05,	1.56E-05]
    #std_thold = np.array([1.36E-05,	8.94E-05, 7.33E-05,	3.54E-05, 1.98E-05,	1.48E-05])
    std_thold = np.array([1.39E-05,	8.38E-05, 7.71E-05,	3.42E-05, 2.14E-05,	1.56E-05])
    std_ave = np.average(std_allfile, axis=0)
    std_pct = std_ave/std_thold *100

    # Separate Thresholds
    std_sep_thold_ave = np.array([[1.07E-05, 1.17E-04, 5.53E-05, 1.24E-05, 1.27E-05, 1.02E-05],
                              [1.53E-05, 7.25E-05, 8.91E-05, 4.35E-05, 1.33E-05, 1.57E-05],
                              [7.38E-06, 1.08E-04, 6.25E-05, 1.31E-05, 1.10E-05, 1.02E-05],
                              [1.91E-05, 6.63E-05, 7.89E-05, 4.44E-05, 1.35E-05, 2.19E-05],
                              [7.20E-06, 9.17E-05, 7.02E-05, 1.48E-05, 3.96E-05, 8.50E-06]])

    std_sep_thold_max = np.array([[1.08E-05, 1.19E-04, 5.55E-05, 1.27E-05, 1.28E-05, 1.04E-05],
                              [1.56E-05, 7.39E-05, 8.96E-05, 4.51E-05, 1.35E-05, 1.80E-05],
                              [7.41E-06, 1.10E-04, 6.32E-05, 1.34E-05, 1.19E-05, 1.03E-05],
                              [1.97E-05, 6.75E-05, 7.91E-05, 4.50E-05, 1.39E-05, 2.48E-05],
                              [7.23E-06, 9.27E-05, 7.07E-05, 1.53E-05, 4.13E-05, 8.73E-06]])

    std_sep_thold_final = np.array([[1.0110E-05,	9.5361E-05,	7.2536E-05,	4.7609E-05,	1.8368E-05,	1.5564E-05],
[1.7314E-05,	6.1918E-05,	7.7077E-05,	4.7609E-05,	1.9421E-05,	2.6943E-05],
[7.2192E-06,	8.9109E-05,	7.0271E-05,	4.7609E-05,	1.9180E-05,	1.5956E-05],
[2.0010E-05,	5.9469E-05,	6.8358E-05,	4.7609E-05,	2.0237E-05,	3.8739E-05],
[7.2113E-06,	7.6864E-05,	7.1004E-05,	4.7609E-05,	9.4813E-05,	1.2832E-05]])
    std_sep_thold = std_sep_thold_final
    std_sep_ave = np.average(std_sep_allfile, axis=0)
    #print('std_sep_ave is', std_sep_ave)
    std_sep_pct = std_sep_ave/std_sep_thold * 100
    #print('separate path check percentage is ', std_sep_pct)


    pct_count = 0
    failJoint = []
    passJoint = []
    for pct in std_pct:
        pct_count += 1
        if pct < 100:
            passJoint.append(np.array([int(pct_count), pct]))
        else:
            failJoint.append(np.array([int(pct_count), pct]))

    if count == 0:
        print("Error! Log files are not found")
        exit()
    print('')
    print('Writing stdev values to files')
    file_result_total = file_result_path +"Stdev_all_runs.txt"
    np.savetxt(file_result_total, std_allfile, delimiter=",", newline="\n")

    file_result_total_pct = file_result_path +"Stdev_all_runs_pct.txt"
    np.savetxt(file_result_total_pct, std_pct, delimiter=",", newline="\n", fmt='%.2f')

    file_result_total_sep = file_result_path + "Stdev_sep.txt"
    np.savetxt(file_result_total_sep, std_sep_ave, delimiter=",", newline="\n")

    file_result_total_sep_indiv1 = file_result_path + "Stdev_sep_indiv1.txt"
    np.savetxt(file_result_total_sep_indiv1, std_sep_allfile[0], delimiter=",", newline="\n")

    file_result_total_sep_indiv2 = file_result_path + "Stdev_sep_indiv2.txt"
    np.savetxt(file_result_total_sep_indiv2, std_sep_allfile[1], delimiter=",", newline="\n")

    file_result_total_sep_indiv3 = file_result_path + "Stdev_sep_indiv3.txt"
    np.savetxt(file_result_total_sep_indiv3, std_sep_allfile[2], delimiter=",", newline="\n")


    file_result_total_sep_pct = file_result_path +"Stdev_sep_pct.txt"
    np.savetxt(file_result_total_sep_pct, std_sep_pct, delimiter=",", newline="\n", fmt='%.2f')
    print('Finished, please check the data folder for output files')
    print('')






    for pct_count, pct in failJoint:
        print('Joint '+np.str(int(pct_count))+' fails with '+np.str('%.2f' %pct)+'% of threshold value')
    for pct_count, pct in passJoint:
        print('Joint '+np.str(int(pct_count))+' passes with '+np.str('%.2f' %pct)+'% of threshold value')




if __name__=="__main__":
    # Define global file_path and filter settings
    file_path = filedialog.askdirectory(initialdir="/", title="Select a Folder")
    cutoff = 0.5/(2*np.pi)
    fs = 1
    order = 6
    str = "Set Jog To mode"
    # Run main program
    main()

