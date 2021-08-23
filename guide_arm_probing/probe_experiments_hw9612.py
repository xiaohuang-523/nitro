import numpy as np
import Readers as Yomiread
import Writers as Yomiwrite
import registration
import plot as Yomiplot
import matplotlib.pyplot as plt

if __name__ == "__main__":
    PROBE_MARKER_GA_FILE = "C:\\Neocis\\Output\\GAPTRegistrationData\\gapt-registration-data\\final_probe_poses.json"
    FIDUCIAL_ARRAY_FS_FILE = "C:\\Neocis\\FiducialArrays\\FXT-0086-07-LRUL-MFG-Splint.txt"
    PROBE_MARKER_CT_FILE = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous" \
                             "\\HW-9612 Point-pair experiments\\Yomiplan Case File\\markers_ct.txt"
    FIDUCIAL_ARRAY_CT_FILE = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous" \
                             "\\HW-9612 Point-pair experiments\\Yomiplan Case File\\fiducials_array_ct.txt"

    # define which fiducial markers are used in registration
    #selected_marker_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    #selected_marker_number = [0, 1, 7, 8, 12, 13]   # 6 i-o-i
    selected_marker_number = [0, 3, 6, 8, 9, 12]

    # prepare probing markers in ct and fiducial frames
    marker_ga = Yomiread.read_YomiSettings(PROBE_MARKER_GA_FILE, str='probe_positions') * 1000  # convert m to mm
    marker_ct = Yomiread.read_csv(PROBE_MARKER_CT_FILE, 4, -1)[:,1:]
    marker_ga_selected = []
    marker_ct_selected = []
    for idx in selected_marker_number:
        marker_ga_selected.append(marker_ga[idx, :])
        marker_ct_selected.append(marker_ct[idx, :])
    marker_ga_selected = np.asarray(marker_ga_selected)
    marker_ct_selected = np.asarray(marker_ct_selected)
    print('marker_ga are', marker_ga_selected)
    print('marker_ct are', marker_ct_selected)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    Yomiplot.plot_3d_points(marker_ct_selected, ax, color='g')
    Yomiplot.plot_3d_points(marker_ga_selected, ax, color='blue')
    plt.show()
    # Point-pair registration (ct = R * ga + t)
    R, t = registration.point_set_registration(marker_ct_selected, marker_ga_selected)
    print('R is', R)
    print('t is', t)

    # Generate fiducial array in image space
    fiducial_array_fs = Yomiread.read_csv_specific_rows(FIDUCIAL_ARRAY_FS_FILE, 4, [3, -1], delimiter=' ')[:,1:]
    fiducial_array_ct = []
    for point in fiducial_array_fs:
        #fiducial_array_ct.append(np.matmul(R, point) + t)
        fiducial_array_ct.append(np.matmul(np.linalg.inv(R), (point-t)))
    fiducial_array_ct = np.asarray(fiducial_array_ct)
    Yomiwrite.write_csv_matrix(FIDUCIAL_ARRAY_CT_FILE, fiducial_array_ct, fmt='%.6f', delim=' ')

    landmark_ct = np.array([45.872546347336275, 47.517325926047029, 52.978035379603348])
    #landmark_fs = np.matmul(np.linalg.inv(R),(landmark_ct - t))
    landmark_fs = np.matmul(R, landmark_ct) + t
    print('landmark_fs is', landmark_fs)
