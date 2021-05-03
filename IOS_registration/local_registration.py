# Local registration is performed based on ICP method.

import open3d as o3d
import numpy as np
import copy

def draw_registration_result_o3d_point_cloud(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # yellow
    # source_temp.paint_uniform_color([0.8, 0, 0.4])      # red
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # blue
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def draw_registration_result_points_array(source, target, transformation):
    source_tem = copy.deepcopy(source)
    target_tem = copy.deepcopy(target)

    source_pc = o3d.PointCloud()
    source_pc.points = o3d.Vector3dVector(source_tem)

    target_pc = o3d.PointCloud()
    target_pc.points = o3d.Vector3dVector(target_tem)

    source_pc.paint_uniform_color([1, 0.706, 0])  # yellow
    # source_temp.paint_uniform_color([0.8, 0, 0.4])      # red
    target_pc.paint_uniform_color([0, 0.651, 0.929])  # blue
    source_pc.transform(transformation)
    o3d.visualization.draw_geometries([source_pc, target_pc])



def preprocess_point_cloud(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    o3d.geometry.estimate_normals(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, source_points, target_points, trans_init):
    source = o3d.PointCloud()
    source.points = o3d.Vector3dVector(source_points)

    target = o3d.PointCloud()
    target.points = o3d.Vector3dVector(target_points)
    source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def registration(voxel_size, threshold, source_points, target_points, trans_init):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(voxel_size, source_points, target_points, trans_init)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)

    # perform ICP registration
    o3d.estimate_normals(source, search_param=o3d.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    o3d.estimate_normals(target, search_param=o3d.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    trans_init = result_ransac.transformation

    # print("Initial alignment")
    evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
    # print(evaluation)

    # print("Apply point-to-point ICP")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # draw_registration_result(source, target, reg_p2p.transformation)

    # second ICP
    trans_init2 = reg_p2p.transformation
    # print("Apply 2nd point-to-point ICP")
    reg_p2p2 = o3d.registration.registration_icp(
        source, target, threshold, trans_init2,
        o3d.registration.TransformationEstimationPointToPoint())
    # draw_registration_result(source, target, reg_p2p2.transformation)

    # third ICP
    trans_init3 = reg_p2p2.transformation
    # print("Apply 3rd point-to-point ICP")
    reg_p2p3 = o3d.registration.registration_icp(
        source, target, threshold, trans_init3,
        o3d.registration.TransformationEstimationPointToPoint())
    # print('checking rms')
    # print(reg_p2p3.inlier_rmse)
    # print(reg_p2p3)
    # print("Transformation 3 is:")
    # print(reg_p2p3.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2p3.transformation)
    return reg_p2p3


def registration_simple_pc(voxel_size, threshold, source_points, target_points, trans_init):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(voxel_size, source_points, target_points, trans_init)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)

    # perform ICP registration
    o3d.estimate_normals(source, search_param=o3d.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    o3d.estimate_normals(target, search_param=o3d.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    trans_init = result_ransac.transformation

    # print("Initial alignment")
    evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
    # print(evaluation)

    # print("Apply point-to-point ICP")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # draw_registration_result(source, target, reg_p2p.transformation)

    # second ICP
    trans_init2 = reg_p2p.transformation
    # print("Apply 2nd point-to-point ICP")
    reg_p2p2 = o3d.registration.registration_icp(
        source, target, threshold, trans_init2,
        o3d.registration.TransformationEstimationPointToPoint())
    # draw_registration_result(source, target, reg_p2p2.transformation)

    # third ICP
    trans_init3 = reg_p2p2.transformation
    # print("Apply 3rd point-to-point ICP")
    reg_p2p3 = o3d.registration.registration_icp(
        source, target, threshold, trans_init3,
        o3d.registration.TransformationEstimationPointToPoint())
    # print('checking rms')
    # print(reg_p2p3.inlier_rmse)
    # print(reg_p2p3)
    # print("Transformation 3 is:")
    # print(reg_p2p3.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2p3.transformation)
    return reg_p2p3


# Local registration is performed on each pair of the teeth in two arches
# The registration is to tranpose and align source arch (IOS) with target arch (CT)
def do_local_registration(TRANS_INIT, THRESHOLD_MOLAR, RMS_THRESHOLD, source_arch, target_arch, DEBUG=1):
    if DEBUG == 1:
        tooth_number = target_arch.tooth_list
    else:
        tooth_number = target_arch.existing_tooth_list  # Use
    print('tooth number is', tooth_number)
    for i in tooth_number:
        rms_init = 1
        count = 1
        voxel_step = 0.02
        voxel_size_molar = 0.3
        print('register tooth', i)
        ICP_tem = []
        ICP_result_tem = []
        while rms_init > RMS_THRESHOLD and count < 4:
            # print('iteration ', count)
            ICP_single = registration(voxel_size_molar, THRESHOLD_MOLAR, source_arch.get_tooth(i).points,
                                      target_arch.get_tooth(i).points, TRANS_INIT)
            rms_init = ICP_single.inlier_rmse
            voxel_size_molar = voxel_size_molar - voxel_step * count
            count += 1
            # print('rms_init is', rms_init)
            ICP_tem.append(ICP_single)
            ICP_result_tem.append(ICP_single.inlier_rmse)
            idx = np.where(ICP_result_tem == np.min(ICP_result_tem))[0][0]
            # print('idx is', idx)
            ICP_single = ICP_tem[idx]

        source_arch.get_tooth(i).ICP = ICP_single
        source_arch.get_tooth(i).local_ICP_transformation = ICP_single.transformation
        target_arch.get_tooth(i).local_ICP_transformation = np.linalg.inv(ICP_single.transformation)

        # for presentation purpose
        #draw_registration_result_points_array(source_arch.get_tooth(i).points, target_arch.get_tooth(i).points,
        #                                      ICP_single.transformation)
        #print('registration rms is', ICP_single.inlier_rmse)

        del ICP_single

