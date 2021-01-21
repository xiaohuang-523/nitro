import numpy as np
import Readers as YomiRead
import plot
import matplotlib.pyplot as plt
from scipy import signal
import fileDialog


def butter_highpass_filter(data, cutoff, fs, order=6):
    ba = butter_highpass(cutoff, fs, order=order)
    filtered = signal.lfilter(ba[0], ba[1], data)
    return filtered


def butter_highpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    ba = signal.butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return ba


# check the idx of the line which contains 'str'
# The output is an array with all idx
# idx[2] is the paused status at pose 2
def check_idx_log_api_mode_pause(log_file_path, str="API_CM_MODE switched to 576"):
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    idx_end = len(FILE) - 1
    idx = []
    for line in range(len(FILE)):
        if str in FILE[line]:
            idx_tem = line
            idx.append(idx_tem)
    # idx.append(idx_end)
    idx = np.asarray(idx)
    return idx


# check the idx of the line which contains 'str'
# The output is an array with all idx
# idx[1] is the start jogging to pose 2.
def check_idx_log_api_mode_start(log_file_path, str="API_CM_MODE switched to 592"):
    # str = "Set Jog To mode"
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    idx_end = len(FILE) - 1
    idx = []
    for line in range(len(FILE)):
        if str in FILE[line]:
            idx_tem = line
            idx.append(idx_tem)
    # idx.append(idx_end)
    idx = np.asarray(idx)
    return idx


# check the idx of the line which contains 'str'
# The output is an array with all idx
# idx[1] is the start jogging to pose 2.
def check_idx_log_api_mode_manual_pause(log_file_path, str="Set pause mode (1)"):
    # str = "Set Jog To mode"
    log_file = open(log_file_path, 'r')
    FILE = log_file.readlines()
    idx_end = len(FILE) - 1
    idx = []
    for line in range(len(FILE)):
        if str in FILE[line]:
            idx_tem = line
            idx.append(idx_tem)
    # idx.append(idx_end)
    idx = np.asarray(idx)
    return idx



if __name__ == '__main__':
    RADIANS_2_DEGREE = 180 / np.pi

    #IDX = np.array([9800, 18600, 26200, 35800, 42200, 47100])
    #JOG_MOTION_N = 5
    #cutoff = 0.5 / (2 * np.pi)
    #fs = 1
    #order = 6

    #file = fileDialog.select_file()
    # file = 'G:\\My Drive\\Project\\FnP\\Jitter Test\\20210114_150404_309_neolib_vs.log'
    file = 'G:\\My Drive\\Project\\FnP\\SDJ\\Logs_1_20_2021\\20210120_154016_825_neolib_vs.log'
    idx_pauses = check_idx_log_api_mode_manual_pause(file)[5:6]
    print('idx is', idx_pauses)
    #idx_end = check_idx_log_api_mode_pause(file)[2:7]
    #print('idx_start is', idx_start)
    #print('idx_end is', idx_end)

    for j in range(len(idx_pauses)):
        #print('solving jog ', j + 1)
        QRA, QRAt, QRD, QRDt, UDPA, UDPD = YomiRead.readLog_JointAngle_FnP(file, idx_pauses[j]-10000, idx_pauses[j]+10000, 21)
        QRAt = np.asarray(QRAt)
        QRA = QRA * RADIANS_2_DEGREE
        QRD = QRD * RADIANS_2_DEGREE
        t_normal = (QRAt - QRAt[0])
        UDPA_Q = UDPA[:, 3:9]
        UDPA_A = UDPA[:, 9:15]
        UDPA_V = UDPA[:, 15:21]
        UDPA_L = UDPA[:,0:2]
        UDPD_Q = UDPD[:, 2:]

        for i in range(QRA.shape[1]):
            fig = plt.figure()
            plt.scatter(range(QRA.shape[0]), QRA[:, i], s=1, label='QRA')
            plt.scatter(range(QRD.shape[0]), QRD[:, i], s=1, label='QRD')
            #plt.scatter(range(QRD.shape[0]), QRA[:, i] - QRD[:, i], s=1, label='QRA-QRD')
            #d_filtered = butter_highpass_filter(QRA[:, i] - QRD[:, i], cutoff, fs, order)
            # plt.scatter(range(QRA.shape[0]), QRA[:, i], s=1, label='QRA')
            # plt.scatter(range(QRD.shape[0]), QRD[:, i], s=1, label='QRD')
            #plt.scatter(range(QRD.shape[0]), d_filtered, s=1, label='QRA-QRD (filtered)')
            plt.title('Pause' + np.str(j+1) +' '+ 'Joint' + np.str(i + 1) +' ' + 'QRA & QRD')
            plt.ylabel('degree')
            plt.xlabel('cycle')
            plt.legend()
            #plt.savefig('G:\\My Drive\\Project\\FnP\\SDJ\\Figures\\QRA\\Pause ' + np.str(j+1) + 'Joint ' + np.str(i + 1) + '.png')
            #plt.close()
            fig2 = plt.figure()
            # plt.scatter(range(UDPA_Q.shape[0]), UDPA_Q[:, i], s=1, label='UDPA_Q')
            plt.scatter(range(UDPA_V.shape[0]), UDPA_V[:, i], s=1, label='UDP actual velocity')
            plt.title('Pause' + np.str(j+1) +' ' + 'Joint ' + np.str(i + 1) +' '+ 'UDP Actual Velocity')
            plt.legend()
            #plt.savefig('G:\\My Drive\\Project\\FnP\\SDJ\\Figures\\UDP\\Velocity_Pause' + np.str(j+1) + 'Joint ' + np.str(i + 1) + '.png')
            #plt.close()
            fig3 = plt.figure()
            plt.scatter(range(UDPA_A.shape[0]), UDPA_A[:, i], s=1, label='UDP actual current')
            plt.title('Pause' + np.str(j+1)+' ' + 'Joint ' + np.str(i + 1) +' ' + 'UDP actual current')
            plt.legend()
            #plt.savefig('G:\\My Drive\\Project\\FnP\\SDJ\\Figures\\UDP\\Current_Pause' + np.str(j+1) + 'Joint ' + np.str(i + 1) + '.png')
            #plt.close()
            fig4 = plt.figure()
            plt.scatter(range(UDPD_Q.shape[0]), UDPA_Q[:, i], s=1, label='UDP actual position')
            plt.scatter(range(UDPD_Q.shape[0]), UDPD_Q[:, i], s=1, label='UDP demanded position')
            plt.title('Pause' + np.str(j+1) +' '+ 'Joint ' + np.str(i + 1) +' '+ 'UDP actual and demanded positions')
            plt.legend()
            #plt.savefig('G:\\My Drive\\Project\\FnP\\SDJ\\Figures\\UDP\\UDP_position_Pause' + np.str(j+1) + 'Joint ' + np.str(i + 1) + '.png')
            #plt.close()
            print('figure saved for Joint ' + np.str(i + 1))

        fig5 = plt.figure()
        plt.scatter(UDPA_L[:,0], UDPA_L[:,1], s=1, label='queue length')
        plt.title('Pause' + np.str(j+1) +' '+ 'MCM queue length')
        plt.legend()
        #plt.savefig('G:\\My Drive\\Project\\FnP\\SDJ\\Figures\\UDP\\Pause' + np.str(j+1) + 'queue_length.png')
        #plt.close()


        del QRA, QRAt, QRD, QRDt, UDPA, UDPA_A, UDPA_Q, UDPA_V, UDPD, UDPD_Q

    plt.show()