# Surface matching algorithm development.
# Curve vs. Surface
# Curve vs. Curve
# Surface vs. Surface
#
# developed for HW-9698
# by Xiao Huang @ 08/17/2021

import point_cloud_manipulation as pcm
import Readers as Yomiread
import open3d as o3d
import numpy as np

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

    # Prepare point cloud object
    voxel_size = 1.5
    #marker_ct_pc = pcm.preprocess_point_cloud(marker_ct, voxel_size)
    marker_ct_pc = o3d.PointCloud()
    marker_ct_pc.points = o3d.Vector3dVector(marker_ct)
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = o3d.geometry.voxel_down_sample(marker_ct_pc, voxel_size)

# Find tangent line of a 3d Curve
# Ideas:
#       1. Average of the adjacent vectors
#

# Find normal of a 3d point cloud
# Ideas:
#       1. Average of the cross product between each pair of the lines meeting at the point
#           https://stackoverflow.com/questions/16195297/computing-normals-at-each-point-in-a-point-cloud
#       2. Use open3d package to find point local normal
#           http://www.open3d.org/docs/0.7.0/tutorial/Basic/pointcloud.html
#
    print("Recompute the normal of the downsampled point cloud")
    o3d.geometry.estimate_normals(marker_ct_pc, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
    o3d.visualization.draw_geometries([marker_ct_pc])

    print("Print a normal vector of the 0th point")
    print(marker_ct_pc.normals[0])
    print("Print the normal vectors of the first 10 points")
    print(np.asarray(marker_ct_pc.normals)[:10, :])
    print("")

    # Offset each point using probe tip radius
    tip_radius = 3
    marker_ct_offset = []
    for i in range(len(marker_ct_pc.points)):
        new_point = marker_ct_pc.points[i] + marker_ct_pc.normals[i] * tip_radius
        marker_ct_offset.append(new_point)
        new_point = marker_ct_pc.points[i] - marker_ct_pc.normals[i] * tip_radius
        marker_ct_offset.append(new_point)

    marker_ct_offset_pc = o3d.PointCloud()
    marker_ct_offset_pc.points = o3d.Vector3dVector(marker_ct_offset)
    o3d.visualization.draw_geometries([marker_ct_offset_pc])


