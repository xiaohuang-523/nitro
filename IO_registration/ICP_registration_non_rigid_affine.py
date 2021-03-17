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


def prepare_dataset(voxel_size, source_points, target_points, trans_init):
    print(":: Load two point clouds and disturb initial pose.")

    source = o3d.PointCloud()
    source.points = o3d.Vector3dVector(source_points)

    target = o3d.PointCloud()
    target.points = o3d.Vector3dVector(target_points)
    #trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
    #                         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))

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
    #draw_registration_result(source_down, target_down,
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
    print('checking rms')
    print(reg_p2p3.inlier_rmse)
    print(reg_p2p3)
    print("Transformation 3 is:")
    print(reg_p2p3.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p3.transformation)
    return reg_p2p3


def combine_img(source, target, transformation):
    extra_p = []
    for point in source:
        tem = np.insert(point, 3, 1)
        # print('tem is', tem)
        extra_tem = np.matmul(transformation, tem)
        extra_tem = extra_tem[0:3]
        extra_p.append(extra_tem)
    target_extra_R1 = np.asarray(extra_p)

    extra_p_source = []
    for point in target:
        tem_t = np.insert(point, 3, 1)
        extra_source_tem = np.matmul(np.linalg.inv(transformation), tem_t)
        extra_source_tem = extra_source_tem[0:3]
        extra_p_source.append(extra_source_tem)
    source_extra_R1 = np.asarray(extra_p_source)

    #new_target = np.vstack((target, target_extra_R1))
    new_target = target_extra_R1
    new_source = np.vstack((source, source_extra_R1))
    new_source = source
    return new_source, new_target


def find_ctl_points(pc):
    # Find centroid and principal axes
    # https://github.com/intel-isl/Open3D/issues/2368
    point_cloud = o3d.PointCloud()
    point_cloud.points = o3d.Vector3dVector(pc)
    mean, matrix = o3d.compute_point_cloud_mean_and_covariance(point_cloud)
    axis = np.linalg.eig(matrix)
    return pc, mean, axis[1]


def down_sampling_pc(pc):
    #print('shape of original pc is', np.shape(pc))
    down_pcd = o3d.geometry.PointCloud()
    down_pcd.points = o3d.Vector3dVector(pc)
    #pc_ds = o3d.geometry.voxel_down_sample(down_pcd, 2)
    pc_ds = o3d.geometry.voxel_down_sample(down_pcd, 1)
    pc_ds_points = pc_ds.points
    #print('shape of downsample pc is', np.shape(pc_ds_points))
    return pc_ds_points
    #return pc


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
        tem_1 = point + (i+1)*delta*x
        tem_2 = point - (i+1)*delta*x
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


def plot_3d_pc(pc, uniform_color = False):
    plot_pc = o3d.PointCloud()
    plot_pc.points = o3d.Vector3dVector(pc)
    if uniform_color == True:
        plot_pc.paint_uniform_color([0, 0.651, 0.929])  # yellow
                                                        # [1, 0.706, 0] blue
    visualization.draw_geometries([plot_pc])

if __name__ == "__main__":
    # Read source and target raw data
    source_file_R1 = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Rmolar1.csv'
    source_file_R2 = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Rmolar2.csv'
    source_file_R3 = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Rmolar3.csv'
    source_file_L1 = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Lmolar1.csv'
    source_file_L2 = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Lmolar2.csv'
    source_file_L3 = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Lmolar3.csv'
    source_file_full = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_full.csv'
    source_file_incisor = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_incisor.csv'

    target_file_R1 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Rmolar1.csv'
    target_file_R2 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Rmolar2.csv'
    target_file_R3 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Rmolar3.csv'
    target_file_L1 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Lmolar1.csv'
    target_file_L2 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Lmolar2.csv'
    target_file_L3 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Lmolar3.csv'
    target_file_cylinder = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_full.csv'
    target_file_incisor = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_incisor.csv'
    target_file_check_cylinder = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_cylinder.csv'

    source_points_R1 = Yomiread.read_csv(source_file_R1, 3, -1)
    source_points_R2 = Yomiread.read_csv(source_file_R2, 3, -1)
    source_points_R3 = Yomiread.read_csv(source_file_R3, 3, -1)
    source_points_L1 = Yomiread.read_csv(source_file_L1, 3, -1)
    source_points_L2 = Yomiread.read_csv(source_file_L2, 3, -1)
    source_points_L3 = Yomiread.read_csv(source_file_L3, 3, -1)
    source_points_full = Yomiread.read_csv(source_file_full, 3, -1)
    source_points_incisor = Yomiread.read_csv(source_file_incisor, 3, -1)

    target_points_R1 = Yomiread.read_csv(target_file_R1, 3, -1)
    target_points_R2 = Yomiread.read_csv(target_file_R2, 3, -1)
    target_points_R3 = Yomiread.read_csv(target_file_R3, 3, -1)
    target_points_L1 = Yomiread.read_csv(target_file_L1, 3, -1)
    target_points_L2 = Yomiread.read_csv(target_file_L2, 3, -1)
    target_points_L3 = Yomiread.read_csv(target_file_L3, 3, -1)
    target_points_cylinder = Yomiread.read_csv(target_file_cylinder, 3, -1)
    target_points_incisor = Yomiread.read_csv(target_file_incisor, 3, -1)
    target_points_check_cylinder = Yomiread.read_csv(target_file_check_cylinder, 3, -1)

    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    dicome_points = np.vstack((source_points_R1, source_points_R2))
    dicome_points = np.vstack((dicome_points, source_points_R3))
    dicome_points = np.vstack((dicome_points, source_points_L1))
    dicome_points = np.vstack((dicome_points, source_points_L2))
    dicome_points = np.vstack((dicome_points, source_points_L3))
    dicome_points = np.vstack((dicome_points, source_points_full))
    dicome_points = np.vstack((dicome_points, source_points_incisor))

    stl_points = np.vstack((target_points_R1, target_points_R2))
    stl_points = np.vstack((stl_points, target_points_R3))
    stl_points = np.vstack((stl_points, target_points_L1))
    stl_points = np.vstack((stl_points, target_points_L2))
    stl_points = np.vstack((stl_points, target_points_L3))
    stl_points = np.vstack((stl_points, target_points_cylinder))
    stl_points = np.vstack((stl_points, target_points_incisor))
    stl_points = np.vstack((stl_points, target_points_check_cylinder))

    # R2  voxel size is 1
    voxel_size_molar = 0.3  # means 5mm for the dataset  (5mm for the backup data)
    threshold_molar = 50 #15
    ICP_result = []
    ICP_R1 = registration(voxel_size_molar, threshold_molar, source_points_R1, target_points_R1, trans_init)
    ICP_R2 = registration(voxel_size_molar, threshold_molar, source_points_R2, target_points_R2, trans_init)
    ICP_R3 = registration(voxel_size_molar, threshold_molar, source_points_R3, target_points_R3, trans_init)
    ICP_L1 = registration(voxel_size_molar, threshold_molar, source_points_L1, target_points_L1, trans_init)
    ICP_L2 = registration(voxel_size_molar, threshold_molar, source_points_L2, target_points_L2, trans_init)
    ICP_L3 = registration(voxel_size_molar, threshold_molar, source_points_L3, target_points_L3, trans_init)
    ICP_incisor = registration(voxel_size_molar, threshold_molar, source_points_incisor, target_points_incisor, trans_init)
    ICP_result.append(ICP_R1)
    ICP_result.append(ICP_R2)
    ICP_result.append(ICP_R3)
    ICP_result.append(ICP_L1)
    ICP_result.append(ICP_L2)
    ICP_result.append(ICP_L3)
    ICP_result.append(ICP_incisor)

    rms = np.array([ICP_R1.inlier_rmse, ICP_R2.inlier_rmse,ICP_R3.inlier_rmse,ICP_L1.inlier_rmse,ICP_L2.inlier_rmse,ICP_L3.inlier_rmse])
    print('rms is', rms)
    rms_threshold = 1  # 1mm threshold
    rms_min = rms_threshold
    n_best = -1
    for i in range(6):
        if ICP_result[i].inlier_rmse < rms_min:
            rms_min = ICP_result[i].inlier_rmse
            n_best = i
        else:
            rms_min = rms_min
    print(np.str(n_best +1) +'th tooth gives the best initial registration with rmse ' + np.str(rms_min))

    if n_best != -1:
        trans_best = ICP_result[n_best].transformation
    else:
        print('initial registration failed')

    # ICP_R1 = registration(voxel_size_molar, threshold_molar, source_points_R1, target_points_R1, trans_init)
    # ICP_R2 = registration(voxel_size_molar, threshold_molar, source_points_R2, target_points_R2, trans_init)
    # ICP_R3 = registration(voxel_size_molar, threshold_molar, source_points_R3, target_points_R3, trans_init)
    # ICP_L1 = registration(voxel_size_molar, threshold_molar, source_points_L1, target_points_L1, trans_init)
    # ICP_L2 = registration(voxel_size_molar, threshold_molar, source_points_L2, target_points_L2, trans_init)
    # ICP_L3 = registration(voxel_size_molar, threshold_molar, source_points_L3, target_points_L3, trans_init)

    source_2teeth = o3d.PointCloud()
    source_2teeth.points = o3d.Vector3dVector(dicome_points)

    target_2teeth = o3d.PointCloud()
    target_2teeth.points = o3d.Vector3dVector(stl_points)

    source_2teeth.paint_uniform_color([0, 0.651, 0.929])  # yellow
    # source_temp.paint_uniform_color([0.8, 0, 0.4])      # red
    target_2teeth.paint_uniform_color([1, 0.706, 0])  # blue

    draw_registration_result(source_2teeth, target_2teeth, trans_best)

    # Find control points
    source_points_R1 = down_sampling_pc(source_points_R1)
    source_points_R2 = down_sampling_pc(source_points_R2)
    source_points_R3 = down_sampling_pc(source_points_R3)
    source_points_L1 = down_sampling_pc(source_points_L1)
    source_points_L2 = down_sampling_pc(source_points_L2)
    source_points_L3 = down_sampling_pc(source_points_L3)
    source_points_incisor = down_sampling_pc(source_points_incisor)

    new_source_points_R1, new_target_points_R1 = combine_img(source_points_R1, target_points_R1, ICP_result[0].transformation)
    new_source_points_R2, new_target_points_R2 = combine_img(source_points_R2, target_points_R2, ICP_result[1].transformation)
    new_source_points_R3, new_target_points_R3 = combine_img(source_points_R3, target_points_R3, ICP_result[2].transformation)
    new_source_points_L1, new_target_points_L1 = combine_img(source_points_L1, target_points_L1, ICP_result[3].transformation)
    new_source_points_L2, new_target_points_L2 = combine_img(source_points_L2, target_points_L2, ICP_result[4].transformation)
    new_source_points_L3, new_target_points_L3 = combine_img(source_points_L3, target_points_L3, ICP_result[5].transformation)
    new_source_points_incisor, new_target_points_incisor = combine_img(source_points_incisor, target_points_incisor, ICP_result[6].transformation)

    # control points using principal axes
    ctl_p_source1, mean_source1, axe_source1 = find_ctl_points(new_source_points_R1)
    ctl_p_source2, mean_source2, axe_source2 = find_ctl_points(new_source_points_R2)
    ctl_p_source3, mean_source3, axe_source3 = find_ctl_points(new_source_points_R3)
    ctl_p_source4, mean_source4, axe_source4 = find_ctl_points(new_source_points_L1)
    ctl_p_source5, mean_source5, axe_source5 = find_ctl_points(new_source_points_L2)
    ctl_p_source6, mean_source6, axe_source6 = find_ctl_points(new_source_points_L3)
    ctl_p_source7, mean_source7, axe_source7 = find_ctl_points(new_source_points_incisor)

    ctl_p_target1, mean_target1, axe_target1 = find_ctl_points(new_target_points_R1)
    ctl_p_target2, mean_target2, axe_target2 = find_ctl_points(new_target_points_R2)
    ctl_p_target3, mean_target3, axe_target3 = find_ctl_points(new_target_points_R3)
    ctl_p_target4, mean_target4, axe_target4 = find_ctl_points(new_target_points_L1)
    ctl_p_target5, mean_target5, axe_target5 = find_ctl_points(new_target_points_L2)
    ctl_p_target6, mean_target6, axe_target6 = find_ctl_points(new_target_points_L3)
    ctl_p_target7, mean_target7, axe_target7 = find_ctl_points(new_target_points_incisor)
    plane_target1 = generate_plate(mean_target1, axe_target1)

    # control points definition end
    ctl_source = np.vstack((ctl_p_source1, ctl_p_source2))
    ctl_source = np.vstack((ctl_source, ctl_p_source3))
    ctl_source = np.vstack((ctl_source, ctl_p_source4))
    ctl_source = np.vstack((ctl_source, ctl_p_source5))
    ctl_source = np.vstack((ctl_source, ctl_p_source6))
    ctl_source = np.vstack((ctl_source, ctl_p_source7))

    ctl_target = np.vstack((ctl_p_target1, ctl_p_target2))
    ctl_target = np.vstack((ctl_target, ctl_p_target3))
    ctl_target = np.vstack((ctl_target, ctl_p_target4))
    ctl_target = np.vstack((ctl_target, ctl_p_target5))
    ctl_target = np.vstack((ctl_target, ctl_p_target6))
    ctl_target = np.vstack((ctl_target, ctl_p_target7))

    # Test affine registraion
    # Step 1 - Prepare initial guess (Use trans_best for translation and rotation) (Use eye matrix for shearing)
    affine_rigid_part = Yomikin.Yomi_parameters(trans_best)
    print('trans_best is', trans_best)
    print('trans_best check is', Yomikin.Yomi_Base_Matrix(affine_rigid_part))
    affine_shear_part = np.zeros(6)
    affine_matrix_init = np.concatenate([affine_rigid_part,affine_shear_part])
    #print('affine_matrix_init is', affine_matrix_init)
    print('affine matrix init check is', affine_registration.get_affine_matrix(affine_matrix_init))

    affine_matrix_optimized = affine_registration.affine_registration(affine_matrix_init, ctl_target, ctl_source)
    print('affine_matrix_optimized is', affine_matrix_optimized)
    trans_final = affine_registration.get_affine_matrix(affine_matrix_optimized)
    print('affine matrix is', trans_final)
    print('best rigid matrix is', trans_best)

    # Check registration of landmarks
    source_ctl = o3d.PointCloud()
    source_ctl.points = o3d.Vector3dVector(ctl_source)

    target_ctl = o3d.PointCloud()
    target_ctl.points = o3d.Vector3dVector(ctl_target)

    source_ctl.paint_uniform_color([0, 0.651, 0.929])  # yellow
    # source_temp.paint_uniform_color([0.8, 0, 0.4])      # red
    target_ctl.paint_uniform_color([1, 0.706, 0])  # blue
    #draw_registration_result(source_ctl, target_ctl, trans_best)
    draw_registration_result(source_ctl, target_ctl, trans_final)

    # Check registration of real image
    stl_points_rigid = transpose_pc(stl_points, trans_best)
    stl_points_affine = transpose_pc(stl_points, trans_final)
    dicom_points_affine = transpose_pc(dicome_points, trans_final)
    dicom_points_rigid = transpose_pc(dicome_points, trans_best)

    source_dicom = o3d.PointCloud()
    source_dicom.points = o3d.Vector3dVector(dicome_points)
    #source_dicom.points = o3d.Vector3dVector(stl_points_rigid)
    target_stl = o3d.PointCloud()
    target_stl.points = o3d.Vector3dVector(stl_points)
    #target_stl.points = o3d.Vector3dVector(stl_points_affine)
    source_dicom.paint_uniform_color([0, 0.651, 0.929])  # blue
    target_stl.paint_uniform_color([1, 0.706, 0])  # yellow
    #draw_registration_result(source_dicom, target_stl, trans_init)
    draw_registration_result(source_dicom, target_stl, trans_best)
    draw_registration_result(source_dicom, target_stl, trans_final)
    #draw_registration_result(source_dicom, target_stl, trans_init)

    stl_rigid = o3d.PointCloud()
    #stl_rigid.points = o3d.Vector3dVector(stl_points_rigid)
    stl_rigid.points = o3d.Vector3dVector(dicom_points_rigid)
    stl_rigid.paint_uniform_color([0, 0.651, 0.929])
    stl_affine = o3d.PointCloud()
    stl_affine.points = o3d.Vector3dVector(dicom_points_affine)
    stl_affine.paint_uniform_color([1, 0.706, 0])
    draw_registration_result(stl_rigid, stl_affine, trans_init)



    print('Deformation check affine registration method')
    plane_target1_affine = transpose_pc(plane_target1, trans_final)
    plane_points = o3d.PointCloud()
    plane_points.points = o3d.Vector3dVector(plane_target1_affine)
    visualization.draw_geometries([plane_points])
    
    print('Deformation check TPS method')
    wrap_plane = tps.thin_plate_spline_warp(ctl_target, ctl_source, plane_target1)
    wrap_plane_pc = o3d.PointCloud()
    wrap_plane_pc.points = o3d.Vector3dVector(wrap_plane)
    visualization.draw_geometries([wrap_plane_pc])
    exit()

    print('perform TPS non-rigid registration')
    #wrap_stl = tps.thin_plate_spline_warp(ctl_target, ctl_source, target_points_cylinder)
    wrap_stl = tps.thin_plate_spline_warp(ctl_target, ctl_source, stl_points) # working case
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
