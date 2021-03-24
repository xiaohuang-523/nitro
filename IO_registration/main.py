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
from stl import mesh
import spline_grid_matching as sp_grid
import scipy
import interpolate_3d_points as i3d


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # yellow
    # source_temp.paint_uniform_color([0.8, 0, 0.4])      # red
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # blue
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


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
    #print(":: Load two point clouds and disturb initial pose.")

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
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
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
    #print(result_ransac)
    # draw_registration_result(source_down, target_down,
    #                         result_ransac.transformation)

    # perform ICP registration
    o3d.estimate_normals(source, search_param=o3d.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    o3d.estimate_normals(target, search_param=o3d.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    trans_init = result_ransac.transformation

    #print("Initial alignment")
    evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
    #print(evaluation)

    #print("Apply point-to-point ICP")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    #print(reg_p2p)
    #print("Transformation is:")
    #print(reg_p2p.transformation)
    #print("")
    # draw_registration_result(source, target, reg_p2p.transformation)

    # second ICP
    trans_init2 = reg_p2p.transformation
    #print("Apply 2nd point-to-point ICP")
    reg_p2p2 = o3d.registration.registration_icp(
        source, target, threshold, trans_init2,
        o3d.registration.TransformationEstimationPointToPoint())
    #print(reg_p2p2)
    #print("Transformation 2 is:")
    #print(reg_p2p2.transformation)
    #print("")
    # draw_registration_result(source, target, reg_p2p2.transformation)

    # third ICP
    trans_init3 = reg_p2p2.transformation
    #print("Apply 3rd point-to-point ICP")
    reg_p2p3 = o3d.registration.registration_icp(
        source, target, threshold, trans_init3,
        o3d.registration.TransformationEstimationPointToPoint())
    #print('checking rms')
    #print(reg_p2p3.inlier_rmse)
    #print(reg_p2p3)
    #print("Transformation 3 is:")
    #print(reg_p2p3.transformation)
    #print("")
    #draw_registration_result(source, target, reg_p2p3.transformation)
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

def get_transformed_data(parameter, trans1, trans2, trans3):
    print('parameter is', parameter)
    print('parameter shape is', np.shape(parameter))
    p1 = transpose_pc(parameter, trans1)
    p2 = transpose_pc(parameter, trans2)
    p3 = transpose_pc(parameter, trans3)
    return p1, p2, p3

def vector_angle(v1, v2):
    angle = np.arccos(np.matmul(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi
    if angle > 180:
        angle1 = 360 - angle
    return angle

def get_angluar_error(a0, a1,a2,a3):
    angle1 = []
    angle2 = []
    angle3 = []
    for i in range(a0.shape[0]):
        angle1.append(vector_angle(a0[i,:], a1[i,:]))
        angle2.append(vector_angle(a0[i, :], a2[i, :]))
        angle3.append(vector_angle(a0[i, :], a3[i, :]))
    return np.asarray(angle1), np.asarray(angle2), np.asarray(angle3)

if __name__ == "__main__":
    trans_init = np.eye(4)
    # R2  voxel size is 1
    voxel_size_molar = 0.3  # means 5mm for the dataset  (5mm for the backup data)
    threshold_molar = 50  # 15
    rms_threshold = 0.2

    # ------- Perform local ICP for each tooth ------- #
    tooth_number = np.asarray(range(32, 16, -1))
    ICP_local = []
    ICP_rms = []
    for i in range(16):
        rms_init = 1
        count = 1
        voxel_step = 0.02
        voxel_size_molar = 0.3
        print('register tooth', 32 - i)
        print('rms_init is', rms_init)
        ICP_tem = []
        ICP_result_tem = []
        while rms_init > rms_threshold and count < 4:
            print('iteration ', count)
            ICP_single = registration(voxel_size_molar, threshold_molar, prepare_data.source_stl_points[i],
                                  prepare_data.target_dicom_points[i], trans_init)
            rms_init = ICP_single.inlier_rmse
            voxel_size_molar = voxel_size_molar - voxel_step * count
            count += 1
            print('rms_init is', rms_init)
            ICP_tem.append(ICP_single)
            ICP_result_tem.append(ICP_single.inlier_rmse)
            idx = np.where(ICP_result_tem == np.min(ICP_result_tem))[0][0]
            print('idx is', idx)
            ICP_single = ICP_tem[idx]

        print('final rmse is', ICP_single.inlier_rmse)
        ICP_local.append(ICP_single)
        ICP_rms.append(ICP_single.inlier_rmse)
        #source_pc = o3d.PointCloud()
        #source_pc.points = o3d.Vector3dVector(prepare_data.source_stl_points[i])
        #target_pc = o3d.PointCloud()
        #target_pc.points = o3d.Vector3dVector(prepare_data.target_dicom_points[i])
        #draw_registration_result(source_pc, target_pc, ICP_single.transformation)
        del ICP_single
    print('ICP_rms is', ICP_rms)
    # ------- Local ICP END ------- #
    # source tooth is stl, in yellow, target tooth is dicom, in blue
    # deform source points to match target

    # Initial registration is the local registration of the first tooth
    trans_init = ICP_local[0].transformation

    # Prepare control points and splines based on feature points
    # Control points are defined using the DICOM points. [down_sampling, mean(centroid), axis[1]]
    # Spline is fitted using the centroids of control point clouds
    source_ctl = [] # source feature points
    target_ctl = [] # target feature points
    for i in range(16):
        source_ctl_tem, target_ctl_tem = prepare_data.find_ctl_points(prepare_data.source_stl_points[i], prepare_data.target_dicom_points[i], ICP_local[i].transformation)
        source_ctl.append(source_ctl_tem)
        target_ctl.append(target_ctl_tem)

    spline_source = np.vstack((source_ctl[0][1], source_ctl[1][1]))
    spline_target = np.vstack((target_ctl[0][1], target_ctl[1][1]))
    for i in range(14):
        spline_source = np.vstack((spline_source, source_ctl[i+2][1]))
        spline_target = np.vstack((spline_target, target_ctl[i+2][1]))
    Yomiwrite.write_csv_matrix("G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_target_points.csv", spline_target)
    Yomiwrite.write_csv_matrix("G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_source_points.csv", spline_source)

    control_list = range(16)  # if all teeth are used as landmarks
    # control_list = [0, 1, 2, 8, 9, 13, 14, 15]  # if specific landmarks are selected

    # land marks definition. Landmarks can be surface points or centroids
    j = 0       # use surface as landmarks
    # j = 1     # use centroid as landmarks

    for i in range(len(control_list)):
        if ICP_rms[i] < 0.3:        # Implement this for outlier rejection
        # if ICP_rms[i] > 0:        # Allow outlier
            if i == 0:
                ctl_source = source_ctl[control_list[i]][j]
                ctl_target = target_ctl[control_list[i]][j]
            else:
                ctl_source = np.vstack((ctl_source, source_ctl[control_list[i]][j]))
                ctl_target = np.vstack((ctl_target, target_ctl[control_list[i]][j]))
    print('shape of control target is', np.shape(ctl_target))
    print('shape of control source is', np.shape(ctl_source))

    # Optimize initial rigid registration
    rigid_init_parameter = Yomikin.Yomi_parameters(trans_init)
    affine_rigid_part = affine_registration.rigid_registration(rigid_init_parameter, ctl_target, ctl_source)
    #affine_rigid_part = np.array([7.96805197e+01, -9.51933487e+01, -1.56383765e+02,  1.60008558e+00, -3.36706455e-02,  5.18630916e-02])
    trans_rigid = Yomikin.Yomi_Base_Matrix(affine_rigid_part)
    spline_rigid = transpose_pc(spline_source, trans_rigid)
    Yomiwrite.write_csv_matrix("G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_rigid_points.csv", spline_rigid)
    source_stl_points_total_rigid = transpose_pc(prepare_data.source_stl_points_total, trans_rigid)



    # Perform curvilinear correction
    # step1 fit spline and add n_number points(slices)
    n_number = 32
    #sample_target, fine_target = sp.spline_interpolation_3d(spline_target, n_number)

    # step2 convert reference spline to target space with optimized rigid transformation  (Method 1)
    # sample_reference, fine_reference = sp.spline_interpolation_3d(spline_source, n_number) # method 1, first generate mesh then transform
    #fine_reference_transpose = np.asarray(fine_reference).transpose()
    #fine_r_in_t_transpose = transpose_pc(fine_reference_transpose, trans_rigid)
    #fine_r_in_t = [fine_r_in_t_transpose[:,0], fine_r_in_t_transpose[:,1], fine_r_in_t_transpose[:,2]]

    # step2 convert reference spline to target space with optimized rigid transformation  (Method 2)
    #sample_rigid, fine_rigid = sp.spline_interpolation_3d(spline_rigid, n_number) # method 2, first transform then generate mesh
    #fine_r_in_t = fine_rigid

    breaks_file_rigid = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_rigid_breaks.csv'
    coe_file_rigid = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_rigid_coefficients.csv'
    breaks_rigid = Yomiread.read_csv(breaks_file_rigid, 16)[0]
    coe_rigid = Yomiread.read_csv(coe_file_rigid, 4)
    fine_rigid = i3d.pp_matlab(breaks_rigid, coe_rigid, 50)
    fine_r_in_t = fine_rigid

    breaks_file_target = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_target_breaks.csv'
    coe_file_target = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_target_coefficients.csv'
    breaks_target = Yomiread.read_csv(breaks_file_target, 16)[0]
    coe_target = Yomiread.read_csv(coe_file_target, 4)
    fine_target = i3d.pp_matlab(breaks_target, coe_target, 50)

    # step3 perform curvilinear correction with only displacement
    displacement_x = fine_target[0] - fine_r_in_t[0]
    displacement_y = fine_target[1] - fine_r_in_t[1]
    displacement_z = fine_target[2] - fine_r_in_t[2]
    displacement = [displacement_x, displacement_y, displacement_z]
    center_t, boundary_t = sp_grid.check_curvilinear_boundary(fine_target)
    center_r_in_t, boundary_r_in_t = sp_grid.check_curvilinear_boundary(fine_r_in_t)

    spline_rigid_cylindrical = sp_grid.convert_cylindrical(spline_rigid, center_r_in_t)
    spline_rigid_moved = sp_grid.displacement(spline_rigid, spline_rigid_cylindrical, boundary_r_in_t, displacement)

    fine_rigid_point = np.array(np.asarray(fine_rigid).transpose())
    print('fine_rigid_point is', np.shape(fine_rigid_point))
    fine_rigid_point_cylindrical = sp_grid.convert_cylindrical(fine_rigid_point, center_r_in_t)
    fine_rigid_point_moved = sp_grid.displacement(fine_rigid_point, fine_rigid_point_cylindrical, boundary_r_in_t, displacement)
    print('fine_rigid_point_moved is', np.shape(fine_rigid_point_moved))

    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')
    # #ax3d.plot(x_knots, y_knots, z_knots, color = 'blue', label = 'knots')
    ax3d.plot(spline_target[:,0], spline_target[:,1], spline_target[:,2], color = 'r', label = 'target spline')
    #ax3d.plot(spline_rigid[:, 0], spline_rigid[:, 1], spline_rigid[:, 2], color = 'b', label='rigid spline')
    ax3d.plot(spline_rigid_moved[:, 0], spline_rigid_moved[:, 1], spline_rigid_moved[:, 2], color='g', label='moved spline')
    ax3d.plot(fine_target[0], fine_target[1], fine_target[2], 'r*')
    #ax3d.plot(fine_rigid[0], fine_rigid[1], fine_rigid[2], color='r', label='rigid')
    ax3d.plot(fine_rigid_point_moved[:,0], fine_rigid_point_moved[:,1], fine_rigid_point_moved[:,2], color='b', label='rigid move')
    # #ax3d.plot(x_fine_d, y_fine_d, z_fine_d, color='black', label='fine_d')
    fig2.show()
    plt.legend()
    plt.show()

    # print('boundary_t is', np.shape(boundary_t))
    # print('displacement is', np.shape(displacement_x))
    # print('displacement is', np.shape(displacement))
    # fig = plt.figure()
    # plt.scatter(range(len(displacement_x)), displacement_x, color='r', label = 'x')
    # plt.scatter(range(len(displacement_y)), displacement_y, color='r', label='y')
    # plt.scatter(range(len(displacement_z)), displacement_z, color='r', label='z')
    # plt.legend()
    # plt.show()

    ctl_r_in_t = transpose_pc(ctl_source, trans_rigid)
    plot_3d_pc(ctl_r_in_t)

    ctl_r_in_t_cylindrical = sp_grid.convert_cylindrical(ctl_r_in_t, center_r_in_t)
    ctl_r_in_t_moved = sp_grid.displacement(ctl_r_in_t, ctl_r_in_t_cylindrical, boundary_r_in_t, displacement)

    pc_curvilinear = o3d.PointCloud()
    pc_curvilinear.points = o3d.Vector3dVector(ctl_r_in_t_moved)
    pc_rigid = o3d.PointCloud()
    pc_rigid.points = o3d.Vector3dVector(ctl_r_in_t)
    pc2 = o3d.PointCloud()
    pc2.points = o3d.Vector3dVector(ctl_target)

    draw_registration_result(pc_curvilinear, pc2, np.eye(4))
    draw_registration_result(pc_rigid, pc2, np.eye(4))
    draw_registration_result(pc_rigid, pc_curvilinear, np.eye(4))


    # Test affine registraion
    # Step 1 - Prepare initial guess (Use trans_best for translation and rotation) (Use eye matrix for shearing)
    affine_shear_part = np.zeros(6)
    affine_matrix_init = np.concatenate([affine_rigid_part, affine_shear_part])
    #affine_matrix_optimized = affine_registration.affine_registration(affine_matrix_init, ctl_target, ctl_source)
    affine_matrix_optimized = np.array([ 8.05293987e+01, -9.65783900e+01, -1.59065082e+02,  1.60310915e+00,
                                        -4.72492991e-02,  3.50372656e-02,  2.34265491e-02, -2.28752196e-03,
                                        7.10544730e-03,  8.48296189e-04, -1.31058275e-02,  1.61407769e-02])

    trans_final = affine_registration.get_affine_matrix(affine_matrix_optimized)
    trans_final_file = 'G:\My Drive\Project\IntraOral Scanner Registration\Results\Accuracy FXT tests\Register_stl\\final_transformation_surface_with_XY_shear.csv'
    Yomiwrite.write_csv_matrix(trans_final_file, trans_final)


    # Check accuracy
    centroid_reference = []
    centroid_target = []
    c_o_r = []
    c_o_t = []
    c_t_in_r = []
    c_r_in_t = []
    axes_r = []
    axes_t = []
    for i in range(16):
        ct_r, ct_t, ct_o_r, ct_o_t, ct_t_in_r, ct_r_in_t, matrix_r, matrix_t = prepare_data.find_centroid_points_both(prepare_data.source_stl_points[i], prepare_data.target_dicom_points[i], ICP_local[i].transformation)
        centroid_reference.append(ct_r)
        centroid_target.append(ct_t)
        c_o_r.append(ct_o_r)
        c_o_t.append(ct_o_t)
        c_t_in_r.append(ct_t_in_r)
        c_r_in_t.append(ct_r_in_t)
        axes_r.append(matrix_r)
        axes_t.append(matrix_t)
    centroid_reference = np.asarray(centroid_reference)
    centroid_target = np.asarray(centroid_target)
    c_o_r = np.asarray(c_o_r)
    c_o_t = np.asarray(c_o_t)
    c_t_in_r = np.asarray(c_t_in_r)
    c_r_in_t = np.asarray(c_r_in_t)

    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')
    # ax3d.plot(x_knots, y_knots, z_knots, 'go')
    ax3d.plot(centroid_reference[:,0], centroid_reference[:,1], centroid_reference[:,2], color ='r', label = 'combined centroid renference')
    ax3d.plot(c_o_r[:,0], c_o_r[:,1], c_o_r[:,2], color = 'g', label = 'original reference')
    ax3d.plot(c_t_in_r[:,0], c_t_in_r[:,1], c_t_in_r[:,2], color = 'b', label = 'target in reference')
    fig2.show()

    fig3 = plt.figure(3)
    ax3d = fig3.add_subplot(111, projection='3d')
    # ax3d.plot(x_knots, y_knots, z_knots, 'go')
    ax3d.plot(centroid_target[:,0], centroid_target[:,1], centroid_target[:,2], color ='r', label = 'combined centroid renference')
    ax3d.plot(c_o_t[:,0], c_o_t[:,1], c_o_t[:,2], color = 'g', label = 'original reference')
    ax3d.plot(c_r_in_t[:,0], c_r_in_t[:,1], c_r_in_t[:,2], color = 'b', label = 'target in reference')
    fig3.show()

    # Spline source and target are the centroids from CT features only
    # Centroid reference and target are the centroids from CT + IOS features
    #centroid_reference = spline_source
    #centroid_target = spline_target

    centroid_affine = transpose_pc(centroid_reference, trans_final)
    centroid_rigid = transpose_pc(centroid_reference, trans_rigid)
    centroid_init = transpose_pc(centroid_reference, trans_init)

    centroid_r_in_t = centroid_rigid
    centroid_r_in_t_cylindrical = sp_grid.convert_cylindrical(centroid_r_in_t, center_r_in_t)
    centroid_r_in_t_moved = sp_grid.displacement(centroid_r_in_t, centroid_r_in_t_cylindrical, boundary_r_in_t, displacement)

    # # get transformed principal axes - Start
    # p_axe1_r = []
    # p_axe2_r = []
    # p_axe3_r = []
    # p_axe1_t = []
    # p_axe2_t = []
    # p_axe3_t = []
    # for i in range(16):
    #     p_axe1_r.append(axes_r[i][0])
    #     p_axe2_r.append(axes_r[i][1])
    #     p_axe3_r.append(axes_r[i][2])
    #     p_axe1_t.append(axes_t[i][0])
    #     p_axe2_t.append(axes_t[i][1])
    #     p_axe3_t.append(axes_t[i][2])
    #
    # p_axe1_target = np.asarray(p_axe1_t)
    # p_axe2_target = np.asarray(p_axe2_t)
    # p_axe3_target = np.asarray(p_axe3_t)
    # p_axe1_initial, p_axe1_rigid, p_axe1_affine = get_transformed_data(np.asarray(p_axe1_r), trans_rigid_init, trans_rigid, trans_final)
    # p_axe2_initial, p_axe2_rigid, p_axe2_affine = get_transformed_data(np.asarray(p_axe2_r), trans_rigid_init, trans_rigid,
    #                                                                    trans_final)
    # p_axe3_initial, p_axe3_rigid, p_axe3_affine = get_transformed_data(np.asarray(p_axe3_r), trans_rigid_init, trans_rigid,
    #                                                                    trans_final)
    # x1_angle_mismatch_init, x1_angle_mismatch_rigid, x1_angle_mismatch_affine = get_angluar_error(p_axe1_target, p_axe1_initial, p_axe1_rigid, p_axe1_affine)
    # x2_angle_mismatch_init, x2_angle_mismatch_rigid, x2_angle_mismatch_affine = get_angluar_error(p_axe2_target,
    #                                                                                               p_axe2_initial,
    #                                                                                               p_axe2_rigid,
    #                                                                                               p_axe2_affine)
    # x3_angle_mismatch_init, x3_angle_mismatch_rigid, x3_angle_mismatch_affine = get_angluar_error(p_axe3_target,
    #                                                                                               p_axe3_initial,
    #                                                                                               p_axe3_rigid,
    #                                                                                               p_axe3_affine)
    #
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\p_axe1_t.csv', p_axe1_target)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\p_axe2_t.csv', p_axe2_target)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\p_axe3_t.csv', p_axe3_target)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\p_axe1_initial.csv', p_axe1_initial)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\p_axe1_rigid.csv', p_axe1_rigid)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\p_axe1_affine.csv', p_axe1_affine)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\p_axe2_initial.csv', p_axe2_initial)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\p_axe2_rigid.csv', p_axe2_rigid)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\p_axe2_affine.csv', p_axe2_affine)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\p_axe3_initial.csv', p_axe3_initial)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\p_axe3_rigid.csv', p_axe3_rigid)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\p_axe3_affine.csv', p_axe3_affine)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\x1_angle_mismatch_init.csv', x1_angle_mismatch_init)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\x1_angle_mismatch_rigid.csv',
    #                           x1_angle_mismatch_rigid)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\x1_angle_mismatch_affine.csv',
    #                           x1_angle_mismatch_affine)
    #
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\x2_angle_mismatch_init.csv', x2_angle_mismatch_init)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\x2_angle_mismatch_rigid.csv',
    #                           x2_angle_mismatch_rigid)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\x2_angle_mismatch_affine.csv',
    #                           x2_angle_mismatch_affine)
    #
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\x3_angle_mismatch_init.csv', x3_angle_mismatch_init)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\x3_angle_mismatch_rigid.csv',
    #                           x3_angle_mismatch_rigid)
    # Yomiwrite.write_csv_array('G:\\My Drive\\Project\\IntraOral Scanner Registration\\x3_angle_mismatch_affine.csv',
    #                           x3_angle_mismatch_affine)
    # # get transformed principal axes - End

    fig4 = plt.figure(4)
    ax3d = fig4.add_subplot(111, projection='3d')
    # ax3d.plot(x_knots, y_knots, z_knots, 'go')
    ax3d.plot(centroid_target[:, 0], centroid_target[:, 1], centroid_target[:, 2], color='r',
              label='combined centroid target')
    ax3d.plot(centroid_affine[:,0], centroid_affine[:,1], centroid_affine[:,2], color ='g', label = 'affine registration')
    ax3d.plot(centroid_r_in_t_moved[:, 0], centroid_r_in_t_moved[:, 1], centroid_r_in_t_moved[:, 2], color='g',
              label='curvilinear correction')
    ax3d.plot(centroid_rigid[:,0], centroid_rigid[:,1], centroid_rigid[:,2], color = 'b', label = 'rigid registration')
    fig4.show()

    initial_mismatch = centroid_init - centroid_target
    rigid_mismatch = centroid_rigid - centroid_target
    affine_mismatch = centroid_affine - centroid_target
    curvilinear_mismatch = centroid_r_in_t_moved - centroid_target
    rigid_mismatch_error = []
    for point in rigid_mismatch:
        rigid_mismatch_error.append(np.linalg.norm(point))
    affine_mismatch_error = []
    for point in affine_mismatch:
        affine_mismatch_error.append(np.linalg.norm(point))
    init_mismatch_error = []
    for point in initial_mismatch:
        init_mismatch_error.append(np.linalg.norm(point))
    curvilinear_mismatch_error = []
    for point in curvilinear_mismatch:
        curvilinear_mismatch_error.append(np.linalg.norm(point))

    fig5 = plt.figure(5)
    plt.scatter(range(len(rigid_mismatch_error)), rigid_mismatch_error, color = 'r', label='rigid registration mismatch')
    plt.scatter(range(len(affine_mismatch_error)), affine_mismatch_error, color='g', label='affine registration mismatch')
    plt.scatter(range(len(init_mismatch_error)), init_mismatch_error, color='b', label='initial registration mismatch')
    plt.scatter(range(len(curvilinear_mismatch_error)), curvilinear_mismatch_error, color='y', label='curvilinear correction mismatch')

    mismatch_rigid_file = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\rigid_mismatch.csv'
    Yomiwrite.write_csv_array(mismatch_rigid_file, rigid_mismatch_error)
    mismatch_affine_file = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\affine_mismatch.csv'
    Yomiwrite.write_csv_array(mismatch_affine_file, affine_mismatch_error)
    mismatch_init_file = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\initial_mismatch.csv'
    Yomiwrite.write_csv_array(mismatch_init_file, init_mismatch_error)
    mismatch_curvilinear_file = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\curvilinear_mismatch.csv'
    Yomiwrite.write_csv_array(mismatch_curvilinear_file, curvilinear_mismatch_error)

    # Plotting for checking principal axes - Start
    # fig6 = plt.figure(6)
    # plt.scatter(range(len(x1_angle_mismatch_rigid)), x1_angle_mismatch_rigid, color = 'r', label='x1 rigid registration error')
    # plt.scatter(range(len(x1_angle_mismatch_affine)), x1_angle_mismatch_affine, color='g', label='x1 affine registration error')
    # plt.scatter(range(len(x1_angle_mismatch_init)), x1_angle_mismatch_init, color='b', label='x1 initial registration error')
    #
    # fig7 = plt.figure(7)
    # plt.scatter(range(len(x2_angle_mismatch_rigid)), x2_angle_mismatch_rigid, color = 'r', label='x2 rigid registration error')
    # plt.scatter(range(len(x2_angle_mismatch_affine)), x2_angle_mismatch_affine, color='g', label='x2 affine registration error')
    # plt.scatter(range(len(x2_angle_mismatch_init)), x2_angle_mismatch_init, color='b', label='x2 initial registration error')
    #
    # fig8 = plt.figure(8)
    # plt.scatter(range(len(x3_angle_mismatch_rigid)), x3_angle_mismatch_rigid, color = 'r', label='x3 rigid registration error')
    # plt.scatter(range(len(x3_angle_mismatch_affine)), x3_angle_mismatch_affine, color='g', label='x3 affine registration error')
    # plt.scatter(range(len(x3_angle_mismatch_init)), x3_angle_mismatch_init, color='b', label='x3 initial registration error')
    # Plotting for checking principal axes - End
    plt.legend()
    plt.show()


    # Check all points
    source_r_in_t_cylindrical = sp_grid.convert_cylindrical(source_stl_points_total_rigid, center_r_in_t)
    source_r_in_t_moved = sp_grid.displacement(source_stl_points_total_rigid, source_r_in_t_cylindrical, boundary_r_in_t, displacement)
    source_r_in_t_pc = o3d.PointCloud()
    source_r_in_t_pc.points = o3d.Vector3dVector(source_r_in_t_moved)
    plot_3d_pc(source_r_in_t_moved)

    source_stl = o3d.PointCloud()
    source_stl.points = o3d.Vector3dVector(prepare_data.source_stl_points_total)
    # source_dicom.points = o3d.Vector3dVector(stl_points_rigid)
    target_dicom = o3d.PointCloud()
    target_dicom.points = o3d.Vector3dVector(prepare_data.target_dicom_points_total)

    draw_registration_result(source_r_in_t_pc, target_dicom, np.eye(4))
    exit()



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
    # stl_points_rigid = transpose_pc(prepare_data.source_stl_points_total, trans_rigid)
    # stl_points_affine = transpose_pc(prepare_data.source_stl_points_total, trans_final)
    # print('checking cylinders')
    # cylinders_affine_p1 = transpose_pc(prepare_data.checking_cylinders_p1, trans_final)
    # cylinders_affine_p2 = transpose_pc(prepare_data.checking_cylinders_p2, trans_final)
    # cylinders_affine_p3 = transpose_pc(prepare_data.checking_cylinders_p3, trans_final)
    # plot_3d_pc(cylinders_affine_p1)
    #
    # print('writing stl')
    # num_triangles = len(cylinders_affine_p1)
    # data = np.zeros(num_triangles, dtype=mesh.Mesh.dtype)
    # for i in range(num_triangles):
    # #     # I did not know how to use numpy-arrays in this case. This was the major roadblock
    # #     # assign vertex co-ordinates to variables to write into mesh
    #     data["vectors"][i] = np.array([cylinders_affine_p1[i,:], cylinders_affine_p2[i,:], cylinders_affine_p3[i,:]])
    # m = mesh.Mesh(data)
    # m.save('G:\\My Drive\\Project\\IntraOral Scanner Registration\\Results\\Accuracy FXT tests\\Register_stl\\trial5.stl')

    print('checking spline')
    spline_source_rigid = transpose_pc(spline_source, trans_rigid)
    spline_source_affine = transpose_pc(spline_source, trans_final)
    #dicom_points_affine = transpose_pc(dicome_points, trans_final)
    #dicom_points_rigid = transpose_pc(dicome_points, trans_best)
    spline_list = [spline_target, spline_source_rigid, spline_source_affine]
    sp.spline_interpolation_3d_multiple(spline_list)
    Yomiwrite.write_csv_matrix(
        "G:\\My Drive\\Project\\IntraOral Scanner Registration\\STL_pc - trial1\\spline_source_points_rigid_transform.csv",
        spline_source_rigid)
    print('write spline points done')

    affine_spline_mismatch = spline_source_affine - spline_target
    rigid_spline_mismatch = spline_source_rigid - spline_target
    affine_spline_mismatch_error = []
    rigid_spline_mismatch_error = []
    for point in affine_spline_mismatch:
        affine_spline_mismatch_error.append(np.linalg.norm(point))
    for point in rigid_spline_mismatch:
        rigid_spline_mismatch_error.append(np.linalg.norm(point))

    fig5 = plt.figure(5)
    plt.scatter(range(len(rigid_spline_mismatch_error)), rigid_spline_mismatch_error, color = 'r', label='rigid registration error')
    plt.scatter(range(len(affine_spline_mismatch_error)), affine_spline_mismatch_error, color='g', label='affine registration error')

    plt.legend()
    plt.show()


    source_stl = o3d.PointCloud()
    source_stl.points = o3d.Vector3dVector(prepare_data.source_stl_points_total)
    # source_dicom.points = o3d.Vector3dVector(stl_points_rigid)
    target_dicom = o3d.PointCloud()
    target_dicom.points = o3d.Vector3dVector(prepare_data.target_dicom_points_total)



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

    error_rigid = estimate_transformation_error(trans_rigid, ctl_target, ctl_source)
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
