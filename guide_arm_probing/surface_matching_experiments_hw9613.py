import numpy as np
import Readers as Yomiread
import Writers as Yomiwrite
import registration
import plot as Yomiplot
import matplotlib.pyplot as plt

# import open3d
from open3d import *
import open3d

# import ICP functions
import ICP

if __name__ == "__main__":
    PROBE_MARKER_GA_FILE = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\" \
                           "HW-9612 Point-pair experiments\\Experiment Log File\\" \
                           "gapt-registration-data-tooth24-take1" \
                           "\\final_probe_poses.json"
    FIDUCIAL_ARRAY_FS_FILE = "C:\\Neocis\\FiducialArrays\\FXT-0086-07-LRUL-MFG-Splint.txt"
    PROBE_MARKER_CT_FILE = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\DICOM_pc\\drill_fixture\\" \
                           "corrected_dicom_points_tooth24.csv"
    FIDUCIAL_ARRAY_CT_FILE = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous" \
                             "\\HW-9612 Point-pair experiments\\Yomiplan Case File\\fiducials_array_ct.txt"

    # prepare probing markers in ct and fiducial frames
    marker_ga = Yomiread.read_YomiSettings(PROBE_MARKER_GA_FILE, str='probe_positions') * 1000  # convert m to mm
    marker_ct = Yomiread.read_csv(PROBE_MARKER_CT_FILE, 3, -1)

    # Prepare for ICP registration
    marker_ga_pcd = open3d.PointCloud()
    marker_ga_pcd.points = open3d.Vector3dVector(marker_ga)
    marker_ct_pcd = open3d.PointCloud()
    marker_ct_pcd.points = open3d.Vector3dVector(marker_ct)

    # Offset each point using probe tip radius
    open3d.geometry.estimate_normals(marker_ct_pcd,
                                  search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    tip_radius = 2.99/2
    marker_ct_offset = []
    for i in range(len(marker_ct_pcd.points)):
        new_point = marker_ct_pcd.points[i] + marker_ct_pcd.normals[i] * tip_radius
        marker_ct_offset.append(new_point)
        new_point = marker_ct_pcd.points[i] - marker_ct_pcd.normals[i] * tip_radius
        marker_ct_offset.append(new_point)

    marker_ct_offset_pc = open3d.PointCloud()
    marker_ct_offset_pc.points = open3d.Vector3dVector(marker_ct_offset)
    open3d.visualization.draw_geometries([marker_ct_offset_pc])

    # Perform ICP registration
    voxel_size = 1.5
    threshold = 1.0
    trans_init = np.eye(4)
    transform = ICP.registration_simple_pc(voxel_size, threshold, marker_ga, marker_ct_offset, trans_init)

    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    #Yomiplot.plot_3d_points(marker_ct, ax, color='g')
    Yomiplot.plot_3d_points(marker_ga, ax, color='blue')
    plt.show()

    exit()

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