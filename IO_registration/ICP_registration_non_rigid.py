# examples/Python/Basic/icp_registration.py

import Readers as Yomiread
import open3d as o3d
import numpy as np
import copy
from open3d import *



def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0, 0.651, 0.929])  # yellow
    #source_temp.paint_uniform_color([0.8, 0, 0.4])      # red
    target_temp.paint_uniform_color([1, 0.706, 0])  # blue


    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    o3d.geometry.estimate_normals(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, source_points, target_points):
    print(":: Load two point clouds and disturb initial pose.")

    source = o3d.PointCloud()
    source.points = o3d.Vector3dVector(source_points)

    target = o3d.PointCloud()
    target.points = o3d.Vector3dVector(target_points)
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result

def iterative_ICP(source, target, trans_init):
    trans_init3 = reg_p2p2.transformation
    print("Apply 3rd point-to-point ICP")
    reg_p2p3 = o3d.registration.registration_icp(
        source, target, threshold, trans_init3,
        o3d.registration.TransformationEstimationPointToPoint())
    print(reg_p2p3)
    print("Transformation 3 is:")
    print(reg_p2p3.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p3.transformation)

# Function: Select the specific tooth based on boundary values from point cloud
# Input:
#       pc:  point cloud
#       bc:  boundary conditions [ x_min, y_min, z_min, x_max, y_max, z_max]
# Output:
#       pc:  point cloud in the corresponding boundary box
def select_tooth(pc, bc):
    point_cloud = []
    for points in pc:
        if bc[0] < points[0] < bc[3]:
            if bc[1] < points[1] < bc[4]:
                if bc[2] < points[2] < bc[5]:
                    point_cloud.append(points)
    return np.asarray(point_cloud)


def registration(voxel_size, threshold, source_points, target_points):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(voxel_size, source_points, target_points)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down,
                             result_ransac.transformation)

    # perform ICP registration
    o3d.estimate_normals(source, search_param=o3d.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    o3d.estimate_normals(target, search_param=o3d.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    trans_init = result_ransac.transformation

    print("Initial alignment")
    evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    #draw_registration_result(source, target, reg_p2p.transformation)

    # second ICP
    trans_init2 = reg_p2p.transformation
    print("Apply 2nd point-to-point ICP")
    reg_p2p2 = o3d.registration.registration_icp(
        source, target, threshold, trans_init2,
        o3d.registration.TransformationEstimationPointToPoint())
    print(reg_p2p2)
    print("Transformation 2 is:")
    print(reg_p2p2.transformation)
    print("")
    #draw_registration_result(source, target, reg_p2p2.transformation)

    # third ICP
    trans_init3 = reg_p2p2.transformation
    print("Apply 3rd point-to-point ICP")
    reg_p2p3 = o3d.registration.registration_icp(
        source, target, threshold, trans_init3,
        o3d.registration.TransformationEstimationPointToPoint())
    print(reg_p2p3)
    print("Transformation 3 is:")
    print(reg_p2p3.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p3.transformation)
    return reg_p2p3.transformation



if __name__ == "__main__":
    # Read source and target raw data
    source_file_3teeth = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_3teeth.csv'
    target_file_3teeth = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_3teeth.csv'
    source_points_3teeth = Yomiread.read_csv(source_file_3teeth, 3, -1)
    target_points_3teeth = Yomiread.read_csv(target_file_3teeth, 3, -1)

    # Select tooth and plot in Dicom
    dicom_molar = np.array([260, 535, 0, 325, 605, 304]) * 0.2  # x and y are swapped, z = 304-z
    dicom_pre_molar = np.array([300, 450, 0, 338, 487, 304]) * 0.2
    dicom_pre_molar_2 = np.array([318, 425, 0, 370, 455, 304]) * 0.2  # 321, 422, 0, 360, 450, 304
    dicom_molar_pc = select_tooth(source_points_3teeth, dicom_molar)
    dicom_pre_molar_pc = select_tooth(source_points_3teeth, dicom_pre_molar)
    dicom_pre_molar_pc_2 = select_tooth(source_points_3teeth, dicom_pre_molar_2)


    pcd_molar_pc = o3d.PointCloud()
    pcd_molar_pc.points = o3d.Vector3dVector(dicom_pre_molar_pc_2)
    visualization.draw_geometries([pcd_molar_pc])

    # Select tooth and plot in Stl
    stl_molar = np.array([5, 42, -41, 19, 50, -25])
    stl_pre_molar = np.array([-9, 44, -33, -2.5, 50, -21])
    stl_pre_molar_2 = np.array([-13, 46, -25, -8.5, 50, -19.5]) # -14, 44, -25, -8, 50, -18
    stl_molar_pc = select_tooth(target_points_3teeth, stl_molar)
    stl_pre_molar_pc = select_tooth(target_points_3teeth, stl_pre_molar)
    stl_pre_molar_pc_2 = select_tooth(target_points_3teeth, stl_pre_molar_2)

    pcd_stl_molar_pc = o3d.PointCloud()
    pcd_stl_molar_pc.points = o3d.Vector3dVector(stl_pre_molar_pc_2)
    visualization.draw_geometries([pcd_stl_molar_pc])

    # for molar
    voxel_size_molar = 2.0  # means 5mm for the dataset  (5mm for the backup data)
    threshold_molar = 20 #15
    ICP_molar_T = registration(voxel_size_molar, threshold_molar, dicom_molar_pc, stl_molar_pc)

    # for premolar
    voxel_size_premolar = 5.0  # means 5mm for the dataset  (5mm for the backup data)
    threshold_premolar = 20  # 15
    ICP_premolar_T = registration(voxel_size_premolar, threshold_premolar, dicom_pre_molar_pc, stl_pre_molar_pc)

    # for 2nd premolar
    voxel_size_premolar_2 = 2.5  # means 5mm for the dataset  (5mm for the backup data)
    threshold_premolar_2 = 20  # 15
    ICP_premolar_T_2 = registration(voxel_size_premolar_2, threshold_premolar_2, dicom_pre_molar_pc_2, stl_pre_molar_pc_2)

    exit()



    source_2teeth = o3d.PointCloud()
    source_2teeth.points = o3d.Vector3dVector(source_points_2teeth)

    target_2teeth = o3d.PointCloud()
    target_2teeth.points = o3d.Vector3dVector(target_points_2teeth)

    source_2teeth.paint_uniform_color([0, 0.651, 0.929])  # yellow
    # source_temp.paint_uniform_color([0.8, 0, 0.4])      # red
    target_2teeth.paint_uniform_color([1, 0.706, 0])  # blue

    draw_registration_result(source_2teeth, target_2teeth, reg_p2p3.transformation)


    # Point to plane ICP
    # print("Apply point-to-plane ICP")
    # reg_p2l = o3d.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.registration.TransformationEstimationPointToPlane())
    # print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2l.transformation)



    #o3d.estimate_normals(source_down, search_param=o3d.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    #o3d.estimate_normals(target_down, search_param=o3d.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    #result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    #reg_p2l = o3d.registration_icp(source_down, downtarget, threshold, trans_init, TransformationEstimationPointToPlane())


    #result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
    #                                 voxel_size)
    #print(result_icp)
    #draw_registration_result(source, target, result_icp.transformation)

