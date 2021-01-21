import numpy as np
# import matplotlib.pyplot as plt
import math
import csv
import Readers
import Kinematics
import os

def registration(fiducials, targets):
    # T is the transform from fiducials to targets
    # in a perfect world, targets[i,:]^t = T * fiducials[i,:]^t
    # ^t in the prior line means transposed
    if (np.size(fiducials, 0) != np.size(targets, 0)) or (np.size(fiducials, 1) != np.size(targets, 1)):
        print('fiducials and targets must be the same size')
        return False

    fiducial = fiducials[:, 0:3]
    target = targets[:, 0:3]
    meanFiducial = np.mean(fiducial, axis=0)
    meanTarget = np.mean(target, axis=0)
    fid = np.zeros((np.size(fiducial, 0), np.size(fiducial, 1)))
    tgt = np.zeros((np.size(fiducial, 0), np.size(fiducial, 1)))

    for j in range(np.size(fiducials, 0)):
        fid[j, :] = fiducial[j, :] - meanFiducial
        tgt[j, :] = target[j, :] - meanTarget

    u, s, vh = np.linalg.svd(np.matmul(np.transpose(tgt), fid), full_matrices=True)
    R = np.matmul(u, vh)
    t = np.array([meanTarget - np.matmul(R, np.transpose(meanFiducial))])
    T = np.concatenate((np.concatenate((R, t.T), axis=1), [[0, 0, 0, 1]]), axis=0)
    return T

def get_components_neocis_conv(transformation_matrix):
    roll = math.atan2(transformation_matrix[2, 1], transformation_matrix[2, 2])
    pitch = math.asin(-transformation_matrix[2, 0])
    yaw = math.atan2(transformation_matrix[1, 0], transformation_matrix[0, 0])

    t_unrot = np.array([[transformation_matrix[0, 3]], [transformation_matrix[1, 3]], [transformation_matrix[2, 3]]])

    R = transformation_matrix[0:3,0:3]

    t_rot = np.matmul(np.linalg.inv(R), t_unrot)

    t_vec = [t_rot[0, 0], t_rot[1, 0], t_rot[2, 0]]

    return t_vec, [roll, pitch, yaw]


# read AtracsysDataAcquisition.log
def read_atracsys_log(file):
    log_file = open(file, 'r')
    FILE = log_file.readlines()
    QT = []
    QT_data = []
    for line in range(0, len(FILE)):
        if "Collected point =" in FILE[line]:
            QT.append(FILE[line][104:])

    for j in range(len(QT)):
        QT_data_t = np.fromstring(QT[j], dtype=float, sep=',')
        QT_data.append(QT_data_t)
    return np.asarray(QT_data)


if __name__ == '__main__':
    # Prepare data
    folder = 'G:/My Drive/Project/FnP/GA calibration/Trial1/'
    camera_raw = []
    GAq = []
    GAfiducial = []
    for file in os.listdir(folder):
        if file.endswith('atracsysDataAcquisition.log'):
            camera_raw = read_atracsys_log(folder + file) / 1000
        if file == 'robot_points.txt':
            GAq = np.genfromtxt(folder + file, delimiter=',')
            print('robot_points sizhe', np.shape(GAq))
        if file == 'faro_points.txt':
            GAfiducial = np.genfromtxt(folder + file, delimiter=',') / 1000

    if any(v is None for v in [camera_raw, GAq, GAfiducial]):
        print('Error, missing data')
        print('Please confirm robot_points, faro_points and atracsys raw data files are present in the folder')
        raise RuntimeError

    GAq = GAq[:, :6]
    GAq = np.concatenate((GAq, np.zeros((np.size(GAq, 0), 1))), axis=1) # needed to get last transform to the rigid body
    fiducial = np.concatenate((camera_raw, np.ones((camera_raw.shape[0], 1))), axis=1)
    target = np.zeros((GAq.shape[0], 4))

    #GAfiducial = GAfiducial[:, 2:]
    #GAfiducial = np.concatenate((GAfiducial, np.ones((np.size(GAfiducial, 0), 1))), axis=1) # needed to multiply with T
    #GAtarget = np.zeros((np.size(GAfiducial, 0), np.size(GAfiducial, 1)))

    # yomi_settings_filename = "YomiSettings.json"
    # kp0 = Readers.read_json(yomi_settings_filename, 'schunk_kdl')
    # kp0 = np.array([0., 0.167, 0.267, math.pi/ 2, 0., 0., 0., 0., 0., math.pi, -math.pi/ 2, 0, 0.35, 0., 0.,
    #                 0., 0., math.pi, 0., 0., 0., 0., math.pi/ 2, 0., 0.305, 0., 0., 0., -math.pi/2, 0., 0., 0.,
    #                 0., 0., math.pi/ 2, 0., 0., 0., 0.108675, math.pi/2, 0., 0])
    # kp0 = np.array([0., 0.167, 0.267, math.pi / 2, 0., 0., 0., 0., 0., math.pi, -math.pi / 2, 0, 0.35, 0., 0.,
    #                 0., 0., math.pi, 0., 0., 0., 0., math.pi / 2, 0., 0.305, 0., 0., 0., -math.pi / 2, 0., 0., 0.,
    #                 0., 0., math.pi / 2, 0., 0., 0., 0.108675, math.pi / 2, 0., 0, -0.08405014474917231,
    #                 0.006028311895204439, 0.1620064412326282, 0, 0, 0])
    #kp0 = np.array([0.0,    0.174,  0.267,  np.pi / 2.0,   0.0,        0.0,
    #                0.0,    0.0,    0.0,    np.pi,         -np.pi / 2.0,  0.0,
    #                0.350,  0.0,    0.0,    0.0,        0.0,        np.pi,
    #                0.0,    0.0,    0.0,    0.0,        np.pi / 2.0,   0.0,
    #                0.305,  0.0,    0.0,    0.0,        -np.pi / 2.0,  0.0,
    #                0.0,    0.0,    0.0,    0.0,        np.pi / 2.0,   0.0,
    #                0.0,    0.0,    0.108675, np.pi / 2.0,  0.0,       0.0])

    # FnP nominal KDL
    kp0 = np.array([-0.219, 0.0, 0.136, 0.0, 0.0, 0.0,
                    0.0, -0.202, 0.0, 0.0, 0.0, -np.pi / 2.0,
                    0.410, 0.0, 0.0, -np.pi / 2.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, -np.pi / 2.0,
                    0.0, -0.405, 0.0, np.pi, 0.0, -np.pi / 2.0,
                    0.0, 0.0, 0.0, np.pi, 0.0, -np.pi / 2.0,
                    0.0, 0.0, 0.085, 0.0, 0.0, 0.0])

    # Schunk nominal KDL
    #kp0 = np.array([0., 0.167, 0.267, math.pi/ 2, 0., 0.,
    #                0., 0., 0., math.pi, -math.pi/ 2, 0,
    #                0.35, 0., 0., 0., 0., math.pi,
    #                0., 0., 0., 0., math.pi/ 2, 0.,
    #                0.305, 0., 0., 0., -math.pi/2, 0.,
    #                0., 0., 0., 0., math.pi/ 2, 0.,
    #                0., 0., 0.108675, math.pi/2, 0., 0])

    #t_sf_cal = np.array([[1, 0, 0, -0.08405527938661735],
    #                     [0, 1, 0, 0.003859167216053853],
    #                     [0, 0, 1, 0.1618654754567274],
    #                     [0, 0, 0, 1]])

    # FnP t_sf_cal
    t_sf_cal = np.array([[1, 0, 0, -0.084741],
                         [0, 1, 0, 0.0015677],
                         [0, 0, 1, 0.16133],
                         [0, 0, 0, 1]])

    for i in range(np.size(GAq, 0)):
        Matrix = np.matmul(Kinematics.forwardKinematics(kp0, GAq[i, :], 'T'), t_sf_cal)
        #GAtarget[i, :] = Matrix[:, 3].T
        target[i, :] = Matrix[:, 3].T

    #print('robot point is', GAtarget)
    #T = registration(GAfiducial, GAtarget)
    #print('T = ')
    #print(T)

    print('robot point is', target)
    T = registration(fiducial, target)
    print('T = ')
    print(T)

    # T is the transformation of camera frame of reference in robot base frame.
    # T converts camera frame to robot frame.
    #

    p = get_components_neocis_conv(T)
    print('p = ')
    print(p)

    GAfiducialTrans = np.zeros((np.size(fiducial, 0), np.size(fiducial, 1)))

    for i in range(np.size(fiducial, 0)):
        fid_raw = np.array(fiducial[i, :]).T
        GAfiducialTrans[i, :] = np.matmul(T, fid_raw).T * 1000
        GAfiducialTrans[i, 3] = 1

    #readFile = open(folder + 'faro_points_raw.txt', newline='')
    readFile = open(folder + 'faro_points.txt', newline='')
    writeFile = open(folder + 'faro_points_trans.txt', 'w', newline='')
    readFileReader = csv.reader(readFile, delimiter=',')
    writeFileWriter = csv.writer(writeFile, delimiter=',')
    i = 0
    for row in readFileReader:
        row[2:] = GAfiducialTrans[i, :3]
        writeFileWriter.writerow(row)
        i += 1

    readFile.close()
    writeFile.close()