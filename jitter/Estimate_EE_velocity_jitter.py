import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy import signal
import os
import functools
import re
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import fftpack
import signal_processing as YomiSP
import Kinematics as YomiKin
import Readers as YomiRead


def plot_laplace(data):
    fig, ax = plt.subplots(1, data.shape[1], figsize=(4.8, 2.4))
    for m in range(data.shape[1]):
        x = data[: , m]
        #x = butter_highpass_filter(np.asarray(x), cutoff, fs, order)
        print('x shape is', np.shape(x))

        #t = [m * 0.005 for m in range(x.shape[0])]
        #print('t is', t)
        #print('t size is', np.shape(t))


        #X = fftpack.fft(x)
        #freqs = fftpack.fftfreq(len(x)) * 200

        #ax[m].stem(freqs, np.abs(X), use_line_collection=True)
        #ax[m].stem(freqs, np.abs(X))
        #ax[m].set_xlabel('Frequency in Hertz [Hz]')
        #ax[m].set_ylabel('Frequency Domain (Spectrum) Magnitude')
        #ax[m].set_xlim(0, 200 / 2)


        freqs, times, Sx = signal.spectrogram(x, fs=200, window='hanning',
                                              nperseg=1024, noverlap=1024 - 100,
                                              detrend=False, scaling='spectrum')

        #f, ax = plt.subplots(figsize=(4.8, 2.4))
        #ax[m].pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
        ax[m].pcolormesh(times, freqs, 10*np.log10(Sx), cmap='viridis')
        ax[m].set_ylabel('Frequency [Hz]')
        ax[m].set_xlabel('Time [s]')
        ax[m].set_ylim([0, 200/2])


def plot_laplace_single(data):
    #fig = plt.figure()
    #for jogn in range(len(data)):
    for jogn in range(4,5):
        for jointn in range(data[jogn].shape[1]):
            fig = plt.figure()
            x = data[jogn][: , jointn]
            print('At jog ' + np.str(jogn) + ' and joint '+ np.str(jointn))
            t = [m * 0.005 for m in range(x.shape[0])]
            X = fftpack.fft(x)
            freqs = fftpack.fftfreq(len(x)) * 200

        #ax[m].stem(freqs, np.abs(X), use_line_collection=True)
            plt.stem(freqs, np.abs(X))
            plt.xlabel('Frequency in Hertz [Hz]')
            plt.ylabel('Frequency Domain (Spectrum) Magnitude')
            plt.title('Joint '+ np.str(jointn + 1) + ' at jog ' + np.str(jogn + 1))
            plt.xlim(0, 200 / 2)


        #freqs, times, Sx = signal.spectrogram(x, fs=200, window='hanning',
        #                                      nperseg=1024, noverlap=1024 - 100,
        #                                      detrend=False, scaling='spectrum')

        #f, ax = plt.subplots(figsize=(4.8, 2.4))
        #ax[m].pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
        #ax[m].set_ylabel('Frequency [kHz]')
        #ax[m].set_xlabel('Time [s]');


# write csv file with data (matrix)
def write_csv_array_append(file_name, data, fmt = '%.4f'):
    f = open(file_name, 'a')
    data = np.asarray(data)
    if len(np.shape(data)) > 1:
        for data_tem in np.asarray(data):
            np.savetxt(f, data_tem.reshape(1, data_tem.shape[0]), delimiter=",", fmt = fmt)
    else:
        np.savetxt(f, data.reshape(1, len(data)), delimiter=',', fmt = fmt)
        #f.write("\n")
    f.close()


# Check correlation between two variables
# Pearson correlation coefficient
def get_pearson(array1, array2):
    result = stats.pearsonr(array1, array2)

    # Return r value
    # value ranges from -1 to 1.
    # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    return result[0]


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
    VRA_list = []
    QRA = []
    QRD = []
    VRA = []
    QRA_line = []
    QRD_line = []
    VRA_line = []

    for line in range(startLine,endLine):
        if "QRA" in FILE[line]:
            QRA_list.append(FILE[line][65:])
            QRA_line.append(line)
        if "QRD" in FILE[line]:
            QRD_list.append(FILE[line][65:])
            QRD_line.append(line)
        if "VRA" in FILE[line]:
            VRA_list.append(FILE[line][65:])
            VRA_line.append(line)

    for j in range(len(QRA_list)):
        QRA_data_t = np.fromstring(QRA_list[j], dtype=float, sep=',')
        QRA.append(QRA_data_t)

    for k in range(len(QRD_list)):
        QRD_data_t = np.fromstring(QRD_list[k], dtype=float, sep=',')
        QRD.append(QRD_data_t)

    for l in range(len(VRA_list)):
        VRA_data_t = np.fromstring(VRA_list[l], dtype=float, sep=',')
        VRA.append(VRA_data_t)

    return np.asarray(QRA), np.asarray(QRD), np.asarray(VRA), np.asarray(QRA_line), np.asarray(QRD_line), np.asarray(VRA_line)


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
        fluct_filtered = YomiSP.butter_highpass_filter(np.asarray(fluct_total)[:,i], cutoff, fs, order)
        std1_tem = np.std(fluct_filtered)
        std_1[i] = std1_tem

    # For each path, apply high pass filter to eah path
    std_sep = np.zeros((5,6))
    for path in range(5):
        for joint in range(joint_n):
            #print('shape of fluct is', np.shape(np.asarray(fluct[path])))
            fluct_sep_filtered = YomiSP.butter_highpass_filter(np.asarray(fluct[path])[:,joint], cutoff, fs, order)
            std_sep_tem = np.std(fluct_sep_filtered)
            std_sep[path,joint] = std_sep_tem

    #print('std_sep is', std_sep)

    return std_1, std_sep, qra, qrd


if __name__=="__main__":
    # Define global file_path and filter settings
    file_path = filedialog.askopenfilename()
    cutoff = 0.5/(2*np.pi)
    fs = 1
    order = 6
    str = "Set Jog To mode"
    RADIANS_TO_MDEG = 180*1000/np.pi

    # Read guidearm kdl
    kdl = YomiRead.readLog_ga(file_path)[ :-6]
    print('kdl is ', np.shape(kdl))
    print('kdl is ', kdl)

    # Check index based on the string
    idx1 = check_idx_log_api_mode_start(file_path)
    idx2 = check_idx_log_api_mode_pause(file_path)
    print('idx 592 is', idx1)
    print('idx 576 is', idx2)

    VRA_filtered = np.zeros(6)
    VRA_filtered_sep = []
    r = np.zeros((5,6))

    # Define end effector velocity variable
    ee_velocity_total = []
    ee_v_linear_total = []
    ee_v_linear_mag_total = []  # end effector linear velocity magnitude ( A list of 5 elements: one for each jog)

    # Loop for each jog motion
    for i in range(5):
        # data is [QRA, QRD, VRA, QRA_line, QRD_line, VRA_line]
        data = readLog_JointAngle(file_path, idx1[i+1], idx2[i+2])
        qra = data[0]
        ee_velocity_jog = []
        ee_v_linear_jog = []
        velocity_true = data[2] * RADIANS_TO_MDEG
        print('running jog ', i+1)
        # Calculate end-effector velocity with jacobian
        for j in range(qra.shape[0]):
            jacobian_tem = YomiKin.solve_jacob(kdl, qra[j,:])

            # end effector velocity linear and rotation
            j_velocity_rad = data[2][j,:]
            ee_velcoity_tem = np.matmul(jacobian_tem, j_velocity_rad)

            # end effector velocity linear
            ee_velocity_linear_tem = ee_velcoity_tem[0:3]

            # append end effector velocity to array
            ee_velocity_jog.append(ee_velcoity_tem)
            ee_v_linear_jog.append(ee_velocity_linear_tem)

        # Append velocity
        ee_velocity_total.append(ee_velocity_jog)
        ee_v_linear_total.append(ee_v_linear_jog)

        # end effector linear velocity magnitude for each jog (An array with nx1 dimension)
        ee_v_linear_mag_jog = np.linalg.norm(ee_v_linear_jog, axis=1)
        ee_v_linear_mag_total.append(ee_v_linear_mag_jog)
        del ee_v_linear_mag_jog, ee_velocity_jog, ee_v_linear_jog

        # Prepare actual time array based on VRA
        actual_time = []
        for n in range(len(data[2][:,0])):
            actual_time.append(n*0.005)
        actual_time = np.asarray(actual_time)

        # Calculate derivative velocity
        #velocity_pos = []
        #for jointn in range(6):
        #    velocity_tem = []
        #    for j in range(qra.shape[0]-1):
        #        velocity_tem.append((qra[j+1,jointn] - qra[j,jointn])/0.005 * RADIANS_TO_MDEG)
        #    velocity_tem = np.asarray(velocity_tem)
        #    velocity_pos.append(velocity_tem)
        #    del velocity_tem
        #velocity_pos = np.asarray(velocity_pos)

    # Calculate acceleration
    a = []  # end effector acceleration (A list of 5 elements: one for each jog)
    for v_jog in ee_v_linear_total:     # for each jog motion (v_jog is an array of velocities in this jog)
        a_jog = []
        v_jog = np.asarray(v_jog)
        for i in range(v_jog.shape[0]):     # for each cycle in the jog
            # Calculate acceleration
            a_tem = np.zeros(3)
            if i == 0:
                a_tem = (v_jog[i,:] - v_jog[i,:]) / 0.005
            else:
                a_tem = (v_jog[i,:] - v_jog[i-1,:]) / 0.005
            a_jog.append(a_tem)
            del a_tem
        a.append(a_jog)
        del a_jog

    a = np.asarray(a)
    a_mag = []
    for a_tem in a:
        a_tem = np.asarray(a_tem)
        a_mag_tem = np.linalg.norm(a_tem, axis=1)
        a_mag.append(a_mag_tem)

    # Plot EE acceleration vs. time
    for k in range(5):
        fig = plt.figure()
        plt.scatter(range(len(a_mag[k])), a_mag[k])
        plt.ylabel('m/s^2')
        plt.title('EE Accerlation plot at Jog ' + np.str(k+1))

    # Plot EE velocity vs. time
    for o in range(5):
        fig = plt.figure()
        plt.scatter(range(len(ee_v_linear_mag_total[o])), ee_v_linear_mag_total[o])
        plt.ylabel('m/s')
        plt.title('EE Velocity plot at Jog ' + np.str(o+1))
        # Apply filter to VRA and prepare data for frequency analysis (Combine all filtered velocities in one variable)
        # Define VRA filtered variable
        #VRA_filtered_tem = np.zeros((data[2].shape[0], 1))
        #for m in range(6):
        #    velocity_true_filtered = YomiSP.butter_highpass_filter(velocity_true[:,m], cutoff, fs, order)
        #    VRA_filtered_tem = np.hstack((VRA_filtered_tem, velocity_true_filtered.reshape((len(velocity_true_filtered),1))))

            # Code for plotting and comparing two velocities
            #figure = plt.figure()
            #plt.plot(actual_time[:-1], velocity_pos_filtered, c='red', label='Actual velocity')
            #plt.plot(actual_time[:-1], velocity_true_filtered, c='blue', label='Derivative velocity')
            #plt.xlabel('Time (secs)')
            #plt.ylabel('Velocity (mDeg/s)')
            #plt.title('Velocity vs. time (Jog '+ np.str(i+1) + ' Joint ' + np.str(m + 1) +')')
            #plt.legend()

            # Get pearson coefficients of two variables
            #r[i,m] = get_pearson(velocity_true_filtered, velocity_pos_filtered)
            #r[i,m] = get_pearson(velocity_true[:-1,m], velocity_pos[m])
        #VRA_filtered = np.vstack((VRA_filtered, VRA_filtered_tem[:,1:]))
        #VRA_filtered_sep.append(VRA_filtered_tem[:,1:])
    #print('shape of VRA_filtered is', np.shape(VRA_filtered))
    #print('shape of VRA_filtered_sep is ', np.shape(VRA_filtered_sep))
    #plot_laplace_single(np.asarray(VRA_filtered_sep))
    #plot_laplace(np.asarray(VRA_filtered))


    #print('correlations of two velocities are', r)
    #result_file = "G:\\My Drive\\Project\\Jitter Analysis - velocity based\\rvalue.txt"
    #write_csv_array_append(result_file, r)
    # Torque unit is motor rated torque/1000
    # motor rated torque = 296*0.001 mNm


    #for i in range(6):
        #figure = plt.figure()
        #plt.scatter(range(len(data[0][:,i])), data[0][:,i], c="r", label="QRA")
        #plt.scatter(range(len(data[1][:,i])), data[1][:,i], c="blue", label="QRD")
        #plt.plot(range(len(data[2][:,i])), data[2][:,i], c='blue', label='Actual Velocity')

        #plt.plot(actual_time[:-1], velocity_pos[i], c='red', label='velocity based on pos')
        #plt.plot(actual_time, data[2][:, i] * RADIANS_TO_MDEG, c='blue', label='Joint ' + np.str(i + 1))


        #plt.xlabel('Time (secs)')
        #plt.ylabel('Velocity (mDeg/s)')
        #plt.title('Actual Joint Velocity')
        #plt.plot(range(len(data[0][:,i])), (data[0][:,i] - data[1][:,i])*200, c='red', label="Position Jitter" )

        #plt.legend()

    # High pass filter < 2Hz


    plt.show()

