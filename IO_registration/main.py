# examples/Python/Basic/icp_registration.py

import Readers as Yomiread
import open3d as o3d
import numpy as np
import copy
from open3d import *
import tsp as tps
import affine_registration
from skimage.feature import peak_local_max
import Kinematics as Yomikin
from matplotlib import pyplot as plt
import prepare_data
import spinle_interpolation as sp
import Writers as Yomiwrite


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # yellow
    # source_temp.paint_uniform_color([0.8, 0, 0.4])      # red
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # blue
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


def prepare_dataset(voxel_size, source_points, target_points, trans_init):
    print(":: Load two point clouds and disturb initial pose.")

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


def registration(voxel_size, threshold, source_points, target_points, trans_init):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(voxel_size, source_points, target_points, trans_init)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    # draw_registration_result(source_down, target_down,
    #                         result_ransac.transformation)

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
    # draw_registration_result(source, target, reg_p2p.transformation)

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
    # draw_registration_result(source, target, reg_p2p2.transformation)

    # third ICP
    trans_init3 = reg_p2p2.transformation
    print("Apply 3rd point-to-point ICP")
    reg_p2p3 = o3d.registration.registration_icp(
        source, target, threshold, trans_init3,
        o3d.registration.TransformationEstimationPointToPoint())
    print('checking rms')
    print(reg_p2p3.inlier_rmse)
    print(reg_p2p3)
    print("Transformation 3 is:")
    print(reg_p2p3.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p3.transformation)
    return reg_p2p3

#
# def combine_img(source, target, transformation):
#     extra_p = []
#     for point in source:
#         tem = np.insert(point, 3, 1)
#         # print('tem is', tem)
#         extra_tem = np.matmul(transformation, tem)
#         extra_tem = extra_tem[0:3]
#         extra_p.append(extra_tem)
#     target_extra_R1 = np.asarray(extra_p)
#
#     extra_p_source = []
#     for point in target:
#         tem_t = np.insert(point, 3, 1)
#         extra_source_tem = np.matmul(np.linalg.inv(transformation), tem_t)
#         extra_source_tem = extra_source_tem[0:3]
#         extra_p_source.append(extra_source_tem)
#     source_extra_R1 = np.asarray(extra_p_source)
#
#     # new_target = np.vstack((target, target_extra_R1))
#     new_target = target_extra_R1
#     new_source = np.vstack((source, source_extra_R1))
#     new_source = source
#     return new_source, new_target
#
#
# def find_ctl_points(pc):
#     # Find centroid and principal axes
#     # https://github.com/intel-isl/Open3D/issues/2368
#     point_cloud = o3d.PointCloud()
#     point_cloud.points = o3d.Vector3dVector(pc)
#     mean, matrix = o3d.compute_point_cloud_mean_and_covariance(point_cloud)
#     axis = np.linalg.eig(matrix)
#     return pc, mean, axis[1]
#
#
# def down_sampling_pc(pc):
#     # print('shape of original pc is', np.shape(pc))
#     down_pcd = o3d.geometry.PointCloud()
#     down_pcd.points = o3d.Vector3dVector(pc)
#     # pc_ds = o3d.geometry.voxel_down_sample(down_pcd, 2)
#     pc_ds = o3d.geometry.voxel_down_sample(down_pcd, 2)
#     pc_ds_points = pc_ds.points
#     # print('shape of downsample pc is', np.shape(pc_ds_points))
#     return pc_ds_points
#     # return pc


def transpose_pc(pc_2b_convert, transformation):
    pc_converted = []
    for point in pc_2b_convert:
        tem = np.insert(point, 3, 1.)
        tem_converted = np.matmul(transformation, tem)[0:3]
        pc_converted.append(tem_converted)
    return np.asarray(pc_converted)


def generate_plate(point, axis):
    z = axis[2]
    y = axis[1]
    x = axis[0]
    n = 50
    delta = 10 / n
    plane = []
    for i in range(n):
        tem_1 = point + (i + 1) * delta * x
        tem_2 = point - (i + 1) * delta * x
        for j in range(n):
            tem_f_1 = tem_1 + j * delta * y
            tem_f_2 = tem_1 - j * delta * y
            tem_f_3 = tem_2 + j * delta * y
            tem_f_4 = tem_2 - j * delta * y
            plane.append(tem_f_1)
            plane.append(tem_f_2)
            plane.append(tem_f_3)
            plane.append(tem_f_4)
    return np.asarray(plane)


def plot_3d_pc(pc, uniform_color=False):
    plot_pc = o3d.PointCloud()
    plot_pc.points = o3d.Vector3dVector(pc)
    if uniform_color == True:
        plot_pc.paint_uniform_color([0, 0.651, 0.929])  # yellow
        # [1, 0.706, 0] blue
    visualization.draw_geometries([plot_pc])


def estimate_transformation_error(transformation, static_img, moving_img):
    # get the number of landmarks
    n_points = static_img.shape[0]
    err = np.zeros(n_points)
    for x in range(n_points):
        # prepare for homo transformation
        static_point_tem = np.insert(static_img[x,:], 3, 1.)
        moving_point_tem = np.insert(moving_img[x,:], 3, 1.)
        eepos = static_point_tem[0:3] - np.matmul(transformation, moving_point_tem)[0:3]
        err[x] = np.linalg.norm(eepos)
    return err

if __name__ == "__main__":
    trans_init = np.eye(4)
    # R2  voxel size is 1
    voxel_size_molar = 0.3  # means 5mm for the dataset  (5mm for the backup data)
    threshold_molar = 50  # 15
    ICP_result = []
    rms = []
    for i in range(7):
        ICP_single_tooth = registration(voxel_size_molar, threshold_molar, prepare_data.stl[i], prepare_data.dicom[i], trans_init)
        ICP_result.append(ICP_single_tooth)

    rms_threshold = 1  # 1mm threshold
    rms_min = rms_threshold
    n_best = -1
    for i in range(6):
        if ICP_result[i].inlier_rmse < rms_min:
            rms_min = ICP_result[i].inlier_rmse
            n_best = i
        else:
            rms_min = rms_min
    print(np.str(n_best + 1) + 'th tooth gives the best initial registration with rmse ' + np.str(rms_min))

    if n_best != -1:
        trans_best = ICP_result[n_best].transformation
    else:
        trans_best = trans_init
        print('initial registration failed')

    # source tooth is stl, in yellow
    # target tooth is dicom, in blue
    # deform source points to match target

    # Check rigid registration
    trans_rigid = trans_best
    source_rigid = o3d.PointCloud()
    source_rigid.points = o3d.Vector3dVector(prepare_data.stl_points)
    target_rigid = o3d.PointCloud()
    target_rigid.points = o3d.Vector3dVector(prepare_data.dicom_points)
    draw_registration_result(source_rigid, target_rigid, trans_rigid)

    #plane_target1 = generate_plate(mean_target1, axe_target1)

    # control points are [down, mean, axis[1]]
    source_ctl = []
    target_ctl = []
    for i in range(7):
        source_ctl_tem, target_ctl_tem = prepare_data.find_ctl_points(prepare_data.stl[i], prepare_data.dicom[i], ICP_result[i].transformation)
        source_ctl.append(source_ctl_tem)
        target_ctl.append(target_ctl_tem)

    spline_source = np.vstack((source_ctl[0][1], source_ctl[1][1]))
    spline_source = np.vstack((spline_source, source_ctl[2][1]))
    spline_source = np.vstack((spline_source, source_ctl[6][1]))
    spline_source = np.vstack((spline_source, source_ctl[5][1]))
    spline_source = np.vstack((spline_source, source_ctl[4][1]))
    spline_source = np.vstack((spline_source, source_ctl[3][1]))

    spline_target = np.vstack((target_ctl[0][1], target_ctl[1][1]))
    spline_target = np.vstack((spline_target, target_ctl[2][1]))
    spline_target = np.vstack((spline_target, target_ctl[6][1]))
    spline_target = np.vstack((spline_target, target_ctl[5][1]))
    spline_target = np.vstack((spline_target, target_ctl[4][1]))
    spline_target = np.vstack((spline_target, target_ctl[3][1]))


    # control points definition end
    ctl_source = np.vstack((source_ctl[0][0], source_ctl[1][0]))
    ctl_source = np.vstack((ctl_source, source_ctl[2][0]))
    ctl_source = np.vstack((ctl_source, source_ctl[3][0]))
    ctl_source = np.vstack((ctl_source, source_ctl[4][0]))
    ctl_source = np.vstack((ctl_source, source_ctl[5][0]))
    ctl_source = np.vstack((ctl_source, source_ctl[6][0]))

    ctl_target = np.vstack((target_ctl[0][0], target_ctl[1][0]))
    ctl_target = np.vstack((ctl_target,target_ctl[2][0]))
    ctl_target = np.vstack((ctl_target, target_ctl[3][0]))
    ctl_target = np.vstack((ctl_target, target_ctl[4][0]))
    ctl_target = np.vstack((ctl_target, target_ctl[5][0]))
    ctl_target = np.vstack((ctl_target, target_ctl[6][0]))

    # Test affine registraion
    # Step 1 - Prepare initial guess (Use trans_best for translation and rotation) (Use eye matrix for shearing)
    affine_rigid_part = Yomikin.Yomi_parameters(trans_rigid)
    affine_shear_part = np.zeros(6)
    affine_matrix_init = np.concatenate([affine_rigid_part, affine_shear_part])
    affine_matrix_optimized = affine_registration.affine_registration(affine_matrix_init, ctl_target, ctl_source)
    trans_final = affine_registration.get_affine_matrix(affine_matrix_optimized)

    # Check registration of landmarks
    source_ctl_pc = o3d.PointCloud()
    source_ctl_pc.points = o3d.Vector3dVector(ctl_source)
    target_ctl_pc = o3d.PointCloud()
    target_ctl_pc.points = o3d.Vector3dVector(ctl_target)
    draw_registration_result(source_ctl_pc, target_ctl_pc, trans_final)

    # Transform accuracy measurement based on trans_final
    print('preparing accuracy measurements')
    source_accuracy_pos = prepare_data.accuracy_raw[:,0:3]
    source_accuracy_axis = prepare_data.accuracy_raw[:,3:6]
    affine_accuracy_pos = []
    affine_accuracy_axis = []
    for i in range(7):
        pos_tem = np.insert(source_accuracy_pos[i,:],3,1)
        pos_affine = np.matmul(trans_final, pos_tem)[0:3]
        affine_accuracy_pos.append(pos_affine)
        axis_point = source_accuracy_pos[i,:] + source_accuracy_axis[i,:]
        axis_tem = np.insert(axis_point,3,1)
        axis_point_affine = np.matmul(trans_final, axis_tem)[0:3]
        axis_affine = axis_point_affine - pos_affine
        axis_affine = axis_affine/np.linalg.norm(axis_affine)
        affine_accuracy_axis.append(axis_affine)
    affine_accuracy = np.hstack((np.asarray(affine_accuracy_pos), np.asarray(affine_accuracy_axis)))
    Yomiwrite.write_csv_matrix("G:\\My Drive\\Project\\IntraOral Scanner Registration\\Results\\Accuracy FXT tests\\Accuracy assessment\\correction_measurements\\full_arch_cylinder_measurement_5.txt", affine_accuracy)
    print('Accuracy measurements solved')

    # Check registration of real image
    stl_points_rigid = transpose_pc(prepare_data.stl_points, trans_rigid)
    stl_points_affine = transpose_pc(prepare_data.stl_points, trans_final)
    spline_source_rigid = transpose_pc(spline_source, trans_rigid)
    spline_source_affine = transpose_pc(spline_source, trans_final)
    #dicom_points_affine = transpose_pc(dicome_points, trans_final)
    #dicom_points_rigid = transpose_pc(dicome_points, trans_best)
    spline_list = [spline_target, spline_source_rigid, spline_source_affine]
    sp.spline_interpolation_3d_multiple(spline_list)
    plt.legend()
    plt.show()


    source_stl = o3d.PointCloud()
    source_stl.points = o3d.Vector3dVector(prepare_data.stl_points)
    # source_dicom.points = o3d.Vector3dVector(stl_points_rigid)
    target_dicom = o3d.PointCloud()
    target_dicom.points = o3d.Vector3dVector(prepare_data.dicom_points)
    # target_stl.points = o3d.Vector3dVector(stl_points_affine)
    #source_dicom.paint_uniform_color([0, 0.651, 0.929])  # blue
    #target_stl.paint_uniform_color([1, 0.706, 0])  # yellow
    # draw_registration_result(source_dicom, target_stl, trans_init)
    draw_registration_result(source_stl, target_dicom, trans_rigid)
    draw_registration_result(source_stl, target_dicom, trans_final)
    # draw_registration_result(source_dicom, target_stl, trans_init)

    stl_rigid = o3d.PointCloud()
    # stl_rigid.points = o3d.Vector3dVector(stl_points_rigid)
    stl_rigid.points = o3d.Vector3dVector(stl_points_rigid)
    #stl_rigid.paint_uniform_color([0, 0.651, 0.929])
    stl_affine = o3d.PointCloud()
    stl_affine.points = o3d.Vector3dVector(stl_points_affine)
    #stl_affine.paint_uniform_color([1, 0.706, 0])
    draw_registration_result(stl_rigid, stl_affine, trans_init)

    error_rigid = estimate_transformation_error(trans_best, ctl_target, ctl_source)
    error_affine = estimate_transformation_error(trans_final, ctl_target, ctl_source)
    print('rigid transformation mean is', np.mean(error_rigid))
    print('affine transformation mean is', np.mean(error_affine))
    print('rigid transformation max is', np.max(error_rigid))
    print('affine transformation max is', np.max(error_affine))
    plt.figure()
    plt.hist(error_affine, bins=100, range=(0, 10))
    plt.title('affine registration errors')
    plt.figure()
    plt.hist(error_rigid, bins=100, range=(0, 10))
    plt.title('rigid registration errors')
    plt.show()



    # prepare plane
    origin = mean_source1
    axe = np.eye(3)
    plane1 = generate_plate(origin, axe)

    print('Deformation check affine registration method')
    plane_target1_affine = transpose_pc(plane_target1, trans_final)
    plot_3d_pc(plane_target1_affine)

    print('Deformation check TPS method')
    wrap_plane = tps.thin_plate_spline_warp(ctl_target, ctl_source, plane_target1)
    plot_3d_pc(wrap_plane)
    exit()

    print('perform TPS non-rigid registration')
    # wrap_stl = tps.thin_plate_spline_warp(ctl_target, ctl_source, target_points_cylinder)
    wrap_stl = tps.thin_plate_spline_warp(ctl_target, ctl_source, stl_points)  # working case
    wrap_plane = tps.thin_plate_spline_warp(ctl_target, ctl_source, plane_target1)
    ctl_target_check = tps.thin_plate_spline_warp(ctl_target, ctl_source, ctl_target)
    source_dicom = o3d.PointCloud()
    source_ctl_2 = o3d.PointCloud()
    source_dicom.points = o3d.Vector3dVector(dicome_points)
    source_ctl_2.points = o3d.Vector3dVector(ctl_source)

    target_stl = o3d.PointCloud()
    target_ctl_2 = o3d.PointCloud()
    target_stl.points = o3d.Vector3dVector(wrap_stl)
    target_ctl_2.points = o3d.Vector3dVector(ctl_target_check)

    source_dicom.paint_uniform_color([0, 0.651, 0.929])  # blue
    source_ctl_2.paint_uniform_color([0, 0.651, 0.929])
    # source_temp.paint_uniform_color([0.8, 0, 0.4])      # red
    target_stl.paint_uniform_color([1, 0.706, 0])  # yellow
    target_ctl_2.paint_uniform_color([1, 0.706, 0])
    draw_registration_result(source_dicom, target_stl, trans_init)
    draw_registration_result(source_ctl_2, target_ctl_2, trans_init)

    pcd_dicom_molar_pc = o3d.PointCloud()
    pcd_dicom_molar_pc.points = o3d.Vector3dVector(new_source_points_R1)
    visualization.draw_geometries([pcd_dicom_molar_pc])

    pcd_stl_molar_pc = o3d.PointCloud()
    pcd_stl_molar_pc.points = o3d.Vector3dVector(new_target_points_R1)
    visualization.draw_geometries([pcd_stl_molar_pc])

    find_ctl_points(new_target_points_R1)

    exit()
