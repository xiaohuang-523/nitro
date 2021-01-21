import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from scipy import signal
import tkinter as tk
from tkinter import filedialog
import fileDialog as YomiFile

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
    fluct_filtered_data = []
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
            fluct_sep_filtered = butter_highpass_filter(np.asarray(fluct[path])[:,joint], cutoff, fs, order)
            std_sep_tem = np.std(fluct_sep_filtered)
            std_sep[path,joint] = std_sep_tem


    return std_1, std_sep, qra, qrd, fluct, fluct_total


def butter_highpass_filter(data, cutoff, fs, order=6):
    ba = butter_highpass(cutoff, fs, order=order)
    filtered = signal.lfilter(ba[0], ba[1], data)
    return filtered


def butter_highpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    ba = signal.butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return ba


def plot_laplace(data):
    fig, ax = plt.subplots(1, data.shape[1], figsize=(4.8, 2.4))
    for m in range(data.shape[1]):
        x = data[: , m]
        #x = butter_highpass_filter(np.asarray(x), cutoff, fs, order)
        #print('x shape is', np.shape(x))

        t = [m * 0.002 for m in range(x.shape[0])]
        #print('t is', t)
        #print('t size is', np.shape(t))


        #X = fftpack.fft(x)
        #freqs = fftpack.fftfreq(len(x)) * f_s

        #ax[m].stem(freqs, np.abs(X), use_line_collection=True)
        #ax[m].set_xlabel('Frequency in Hertz [Hz]')
        #ax[m].set_ylabel('Frequency Domain (Spectrum) Magnitude')
        #ax[m].set_xlim(0, f_s / 2)

        freqs, times, Sx = signal.spectrogram(x, fs=f_s, window='hanning',
                                              nperseg=1024, noverlap=1024 - 100,
                                              detrend=False, scaling='spectrum')

        #f, ax = plt.subplots(figsize=(4.8, 2.4))
        ax[m].pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
        ax[m].set_ylabel('Frequency [kHz]')
        ax[m].set_xlabel('Time [s]');



#file_name = filedialog.askopenfilename(initialdir="/", title="Select a File")
#file_name = "G:\\My Drive\\Project\\Jittering Analysis\\Log Files_all\\Neocis Test\\BBCX1860\\run2_neolib_vs.log"
#file_name = "G:\\My Drive\\Project\\Jittering Analysis\\Log Files_all\\Neocis Test\\BBCX1858\\20200617_141516_350_neolib_vs.log"
#file_name = "G:\\My Drive\\Project\\Jittering Analysis\\Log Files_all\\Schunk Test\\BBCX1858\\BBCX1858_run2_neolib_vs.log"
#file_name = "G:\\My Drive\\Project\\Jittering Analysis\\Log Files_all\\Schunk Test\\BBRZ4713\\20200414_093247_215_neolib_vs.log"
#file_name = "G:\\My Drive\\Project\\Jittering Analysis\\Inspection log files\\New folder\\BBDR9284 1st run\\BBDR9284 1st run\\log files\\20201023_085933_483_neolib_vs.log"
#file_name = "G:\\My Drive\\Project\\Jittering Analysis\\Log Files_all\\Neocis Test\\BBDR4226 - 10-09-2020 by Ed (Fail on J4)\\BBDR4226 1st and 2nd run Fail data\\BBDR4266 1st run\\20201009_071756_524_neolib_vs.log"

file_name = YomiFile.select_file()

cutoff = 0.5/(2*np.pi)
fs = 1
order = 6
str = "Set Jog To mode"
f_s = 500  # Sampling rate, or number of measurements per second


data = analyze_jittering_single_total_log(file_name, str, cutoff, fs, order)
print('shape of data is', np.shape(data[5]))
#plot_laplace(np.asarray(data[4][1]))
plot_laplace(np.asarray(data[5]))
#x = data[4][0][:,0]
# x = butter_highpass_filter(np.asarray(x), cutoff, fs, order)
# print('x shape is', np.shape(x))
#
# t = [m*0.002 for m in range(x.shape[0])]
# print('t is', t)
# print('t size is', np.shape(t))
# fig, ax = plt.subplots(1, 2)
# ax[0].plot(t, x)
# ax[0].set_xlabel('Time [s]')
# ax[0].set_ylabel('Signal amplitude')
#
#
# X = fftpack.fft(x)
# freqs = fftpack.fftfreq(len(x)) * f_s
#
#
# ax[1].stem(freqs, np.abs(X), use_line_collection=True)
# ax[1].set_xlabel('Frequency in Hertz [Hz]')
# ax[1].set_ylabel('Frequency Domain (Spectrum) Magnitude')
# ax[1].set_xlim(0, f_s/2)
#ax[1].set_ylim(-5, 110)

plt.show()