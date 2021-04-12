import numpy as np
import feature_extraction as fe
import Readers as Yomiread
import local_registration as local_ICP
import open3d as o3d
import copy
import matplotlib.pyplot as plt
from scipy import interpolate
import spline_correction as sc
import Writers as Yomiwrite
import interpolate_3d_points as i3d
import spline_grid_matching as sp_grid
import coordinates
import Kinematics as Yomikin
import affine_registration


def transpose_pc(pc_2b_convert, transformation):
    pc_converted = []
    for point in pc_2b_convert:
        tem = np.insert(point, 3, 1.)
        tem_converted = np.matmul(transformation, tem)[0:3]
        pc_converted.append(tem_converted)
    return np.asarray(pc_converted)


def combine_pc(list):
    combined_pc = list[0]
    for i in range(len(list) - 1):
        # combined_pc = np.vstack((combined_pc, list[i+1]))
        combined_pc = np.concatenate((combined_pc, list[i + 1]))
    return combined_pc


def plot_3d_pc(pc, uniform_color=False):
    plot_pc = o3d.PointCloud()
    plot_pc.points = o3d.Vector3dVector(pc)
    if uniform_color == True:
        plot_pc.paint_uniform_color([0, 0.651, 0.929])  # yellow
        # [1, 0.706, 0] blue
    o3d.visualization.draw_geometries([plot_pc])


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


def convert_cylindrical(pc, center):
    theta = []
    r = []
    z = []
    for point in pc:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        angle = np.arctan2(dy, dx)
        theta.append(angle)
        r.append(np.sqrt(dx ** 2 + dy ** 2))
        z.append(point[2])
    pc_cylindrical = np.asarray([r, theta, z])
    return pc_cylindrical.transpose()


if __name__ == '__main__':

    # ---- Define global constants
    TRANS_INIT = np.eye(4)
    VOXEL_SIZE_ICP = 0.3  # means 5mm for the dataset  (5mm for the backup data)
    THRESHOLD_ICP = 50  # 15
    RMS_LOCAL_REGISTRATION = 0.2
    TOOTH_NUMBER = np.asarray(range(32, 16, -1))
    MISSING_TOOTH_NUMBER = [19, 20, 21]
    ORIGINAL_NO_MISSING = []
    # MISSING_TOOTH_NUMBER = []
    FXT_ALIGN_TOOTH = [20]  # middle tooth in missing tooth number
    NEIGHBOUR_TOOTH = [18]
    NUMBER_SPLINE_GUIDED_POINTS = 100

    MISSING_TOOTH_NUMBER_STEP2 = [17, 18, 19, 20, 21]
    NEIGHBOUR_TOOTH_STEP2 = [18, 22]

    # ---- Define global file paths
    RESULT_TEM_BASE = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\Point Cloud and Registration TEMP\\"

    # ---- Extract features
    source_stl_base = 'G:\My Drive\Project\IntraOral Scanner Registration\STL_pc\\'
    target_dicom_base = 'G:\My Drive\Project\IntraOral Scanner Registration\Point Cloud and Registration Raw Data\DICOM_pc\\'
    stl_file_name = 'new_stl_points_tooth'
    dicom_file_name = 'new_dicom_points_tooth'

    arch_ct_original = fe.Fixture(MISSING_TOOTH_NUMBER, 'CT', 'CT')
    arch_ios = fe.FullArch(MISSING_TOOTH_NUMBER, 'IOS', 'IOS')

    arch_ct = fe.FullArch(MISSING_TOOTH_NUMBER, 'CT', 'CT')
    # convert ct features to IOS space with rigid transformation
    arch_ct_to_ios = fe.FullArch(MISSING_TOOTH_NUMBER, 'CT', 'IOS')
    arch_ct_to_ios_fxt = fe.Fixture(MISSING_TOOTH_NUMBER, 'CT', 'IOS')
    # generate virtual landmarks using ct features with local ICP
    arch_ct_in_ios = fe.FullArch(MISSING_TOOTH_NUMBER, 'CT', 'IOS')
    arch_ct_in_ios_fxt = fe.Fixture(MISSING_TOOTH_NUMBER, 'CT', 'IOS')
    # arch after curvilinear correction - original ios arch
    arch_ios_curvilinear_correction = fe.FullArch(MISSING_TOOTH_NUMBER, 'IOS', 'IOS')
    arch_ios_fxt_curvilinear_correction = fe.Fixture(MISSING_TOOTH_NUMBER, 'IOS', 'IOS')
    # arch after curvilinear correction - virtual ct features
    arch_ct_in_ios_curvilinear_correction = fe.FullArch(MISSING_TOOTH_NUMBER, 'CT', 'IOS')
    arch_ct_in_ios_fxt_curvilinear_correction = fe.Fixture(MISSING_TOOTH_NUMBER, 'CT', 'IOS')

    for i in TOOTH_NUMBER:
        print('reading tooth ', i)
        new_dicom_path = target_dicom_base + dicom_file_name + np.str(i) + '.csv'
        new_stl_path = source_stl_base + stl_file_name + np.str(i) + '.csv'

        stl_tem = Yomiread.read_csv(new_stl_path, 3, -1)
        dicom_tem = Yomiread.read_csv(new_dicom_path, 3, -1)

        tooth_stl_tem = fe.ToothFeature(stl_tem, i, 'IOS', 'IOS')
        arch_ios.add_tooth(i, tooth_stl_tem)

        tooth_dicom_tem = fe.ToothFeature(dicom_tem, i, 'CT', 'CT')
        arch_ct.add_tooth(i, tooth_dicom_tem)
        arch_ct_original.add_tooth(i, tooth_dicom_tem)
        del stl_tem, dicom_tem

    # ---- Perform Local ICP
    local_ICP.do_local_registration(TRANS_INIT, THRESHOLD_ICP, RMS_LOCAL_REGISTRATION, arch_ios, arch_ct)
    # check local registration quality (rms of local ICP of each tooth)
    for i in arch_ios.tooth_list:  # used for algorithm verification (count missing tooth as well)
        # for i in arch_ios.existing_tooth_list: # used for real 3 images processing
        print('tooth ' + np.str(i) + ' ICP rms is ' + np.str(arch_ios.get_tooth(i).ICP.inlier_rmse))

    # Update spline points
    arch_ios.update_spline()
    arch_ct.update_spline()
    arch_ct_original.update_spline()

    # ---- Perform 1st step: register Image3 to Image2
    trans_init_fxt = arch_ct.get_tooth(FXT_ALIGN_TOOTH).local_ICP_transformation
    # Transform ct features to ios space
    for i in arch_ct.tooth_list:
        print('transform tooth ', i)
        points_rigid_tem = transpose_pc(arch_ct_original.get_tooth(i).points, trans_init_fxt)
        points_virtual_tem = transpose_pc(arch_ct_original.get_tooth(i).points, arch_ct_original.get_tooth(i).local_ICP_transformation)
        tooth_feature_rigid_tem = fe.ToothFeature(points_rigid_tem, i, 'CT', 'IOS')
        tooth_feature_virtual_tem = fe.ToothFeature(points_virtual_tem, i, 'CT', 'IOS')
        arch_ct_to_ios_fxt.add_tooth(i, tooth_feature_rigid_tem)
        arch_ct_in_ios_fxt.add_tooth(i, tooth_feature_virtual_tem)
        del points_rigid_tem, tooth_feature_rigid_tem, points_virtual_tem, tooth_feature_virtual_tem

    # Update spline points
    arch_ct_in_ios_fxt.update_spline(fine_flag=True)
    arch_ct_to_ios_fxt.update_spline(fine_flag=True)
    displacement = arch_ct_to_ios_fxt.spline_points_fine - arch_ct_in_ios_fxt.spline_points_fine

    for i in arch_ct_to_ios_fxt.tooth_list:  # for tooth in target
        candidate_tooth = arch_ct_in_ios_fxt.get_tooth(i).points
        candidate_tooth_cylindrical = coordinates.convert_cylindrical(candidate_tooth,
                                                                      arch_ct_in_ios_fxt.spline_points_cylindrical_center)
        corrected_tooth = sc.displacement(candidate_tooth, candidate_tooth_cylindrical,
                                          arch_ct_in_ios_fxt.spline_points_fine_cylindrical_mid_points, displacement)
        corrected_tooth_feature = fe.ToothFeature(corrected_tooth, i, 'CT', 'IOS')
        arch_ct_in_ios_fxt_curvilinear_correction.add_tooth(i, corrected_tooth_feature)

        candidate2_tooth = arch_ios.get_tooth(i).points
        candidate2_tooth_cylindrical = coordinates.convert_cylindrical(candidate2_tooth,
                                                                       arch_ct_in_ios_fxt.spline_points_cylindrical_center)
        corrected2_tooth = sc.displacement(candidate2_tooth, candidate2_tooth_cylindrical,
                                           arch_ct_in_ios_fxt.spline_points_fine_cylindrical_mid_points, displacement)
        corrected2_tooth_feature = fe.ToothFeature(corrected2_tooth, i, 'IOS', 'IOS')
        arch_ios_curvilinear_correction.add_tooth(i, corrected2_tooth_feature)

        del candidate_tooth, candidate2_tooth

    arch_ios_curvilinear_correction.update_spline()
    arch_ct_in_ios_fxt_curvilinear_correction.update_spline()

    correction_error = []
    corrected_spline = []  # spline after curvilinear correction
    correction_spline = []  # spline before curvilinear correction
    original_full_spline = []  # spline of arch_ct_to_ios
    for i in arch_ct_to_ios_fxt.tooth_list:
        error = arch_ct_to_ios_fxt.get_spline_points(i) - arch_ct_in_ios_fxt_curvilinear_correction.get_spline_points(i)
        correction_error.append(np.linalg.norm(error))
        original_full_spline.append(arch_ct_to_ios_fxt.get_tooth(i).centroid)
        # original_spline.append(arch_ct_to_ios.get_spline_points(i))
        corrected_spline.append(arch_ct_in_ios_fxt_curvilinear_correction.get_spline_points(i))
        # correction_spline.append(arch_ct_in_ios.get_spline_points(i))
    corrected_spline = np.asarray(corrected_spline)
    # original_spline = np.asarray(original_spline)
    # correction_spline = np.asarray(correction_spline)

    original_full_spline = np.asarray(original_full_spline)
    fig1 = plt.figure()
    plt.scatter(range(len(correction_error)), correction_error)

    fig2 = plt.figure()
    plt.scatter(original_full_spline[:, 0], original_full_spline[:, 1], label='original full spline points',
                color='red')
    plt.plot(arch_ct_to_ios_fxt.spline_points_fine[:, 0], arch_ct_to_ios_fxt.spline_points_fine[:, 1], '-',
             label='spline with missing teeth',
             color='green')
    plt.scatter(corrected_spline[:, 0], corrected_spline[:, 1], label='corrected spline', color='blue')
    # plt.plot(test[:,0], test[:,1], label='test')
    plt.legend()
    plt.show()

    #exit()
    arch_ios_2 = fe.FullArch(MISSING_TOOTH_NUMBER_STEP2, 'IOS', 'IOS')  # the ios arch after 1st step correction
    arch_ct_to_ios = fe.FullArch(MISSING_TOOTH_NUMBER_STEP2, 'CT', 'IOS')

    for i in TOOTH_NUMBER:
        arch_ios_2.add_tooth(i, arch_ct_in_ios_fxt_curvilinear_correction.get_tooth(i))

    # Update spline points
    arch_ios_2.update_spline()

    ctl_target = []
    ctl_source = []
    for i in NEIGHBOUR_TOOTH_STEP2:
        ctl_target.append(arch_ct_to_ios_fxt.get_tooth(i).points)
        ctl_source.append(arch_ios_2.get_tooth(i).points)
    ctl_source = combine_pc(ctl_source)
    ctl_target = combine_pc(ctl_target)

    # ---- Perform 2st step: register Image1 to modified Image2
    #trans_init = arch_ct.get_tooth(NEIGHBOUR_TOOTH_STEP2[1]).local_ICP_transformation
    trans_init = np.eye(4)
    rigid_init_parameter = Yomikin.Yomi_parameters(trans_init)
    affine_rigid_part = affine_registration.rigid_registration(rigid_init_parameter, ctl_target, ctl_source)
    trans_rigid = Yomikin.Yomi_Base_Matrix(affine_rigid_part)
    # Transform ct features to ios space
    for i in arch_ios_2.tooth_list:
        print('transform tooth ', i)
        points_rigid_tem = transpose_pc(arch_ios_2.get_tooth(i).points, trans_rigid)
        #points_virtual_tem = transpose_pc(arch_ct.get_tooth(i).points, arch_ct.get_tooth(i).local_ICP_transformation)
        tooth_feature_rigid_tem = fe.ToothFeature(points_rigid_tem, i, 'CT', 'IOS')
        #tooth_feature_virtual_tem = fe.ToothFeature(points_virtual_tem, i, 'CT', 'IOS')
        arch_ct_to_ios.add_tooth(i, tooth_feature_rigid_tem)
        #arch_ct_in_ios.add_tooth(i, tooth_feature_virtual_tem)
        del points_rigid_tem, tooth_feature_rigid_tem #points_virtual_tem, tooth_feature_virtual_tem

    arch_ct_in_ios = arch_ios_2

    # Update spline points
    arch_ct_in_ios.update_spline(fine_flag=True)
    arch_ct_to_ios.update_spline(fine_flag=True)

    print('displacement check', arch_ct_to_ios.spline_points - arch_ct_in_ios.spline_points)
    print('original spline is', arch_ct_in_ios.spline_points)
    print('target spline is', arch_ct_to_ios.spline_points)
    displacement = arch_ct_to_ios.spline_points_fine - arch_ct_in_ios.spline_points_fine
    corrected_spline = sc.displacement_partial(arch_ct_in_ios.spline_points, arch_ct_in_ios.spline_points_cylindrical,
                                       arch_ct_in_ios.spline_points_fine_cylindrical_mid_points, displacement)
    print('corrected spline is', corrected_spline)

    for i in arch_ct_to_ios.tooth_list:
        candidate_tooth = arch_ct_in_ios.get_tooth(i).points
        candidate_tooth_cylindrical = coordinates.convert_cylindrical(candidate_tooth,
                                                                      arch_ct_in_ios.spline_points_cylindrical_center)
        corrected_tooth = sc.displacement_partial(candidate_tooth, candidate_tooth_cylindrical,
                                          arch_ct_in_ios.spline_points_fine_cylindrical_mid_points, displacement)
        corrected_tooth_feature = fe.ToothFeature(corrected_tooth, i, 'CT', 'IOS')
        arch_ct_in_ios_curvilinear_correction.add_tooth(i, corrected_tooth_feature)

        candidate2_tooth = arch_ios.get_tooth(i).points
        candidate2_tooth_cylindrical = coordinates.convert_cylindrical(candidate2_tooth,
                                                                       arch_ct_in_ios.spline_points_cylindrical_center)
        corrected2_tooth = sc.displacement_partial(candidate2_tooth, candidate2_tooth_cylindrical,
                                           arch_ct_in_ios.spline_points_fine_cylindrical_mid_points, displacement)
        corrected2_tooth_feature = fe.ToothFeature(corrected2_tooth, i, 'IOS', 'IOS')
        arch_ios_curvilinear_correction.add_tooth(i, corrected2_tooth_feature)

        del candidate_tooth, candidate2_tooth

    arch_ios_curvilinear_correction.update_spline()
    arch_ct_in_ios_curvilinear_correction.update_spline()

    correction_error = []
    corrected_spline = []  # spline after curvilinear correction
    correction_spline = []  # spline before curvilinear correction
    original_full_spline = []  # spline of arch_ct_to_ios
    for i in arch_ct_to_ios.tooth_list:
        error = arch_ct_to_ios.get_spline_points(i) - arch_ct_in_ios_curvilinear_correction.get_spline_points(i)
        correction_error.append(np.linalg.norm(error))
        original_full_spline.append(arch_ct_to_ios.get_tooth(i).centroid)
        # original_spline.append(arch_ct_to_ios.get_spline_points(i))
        corrected_spline.append(arch_ct_in_ios_curvilinear_correction.get_spline_points(i))
        # correction_spline.append(arch_ct_in_ios.get_spline_points(i))
    corrected_spline = np.asarray(corrected_spline)
    # original_spline = np.asarray(original_spline)
    # correction_spline = np.asarray(correction_spline)

    original_full_spline = np.asarray(original_full_spline)
    fig1 = plt.figure()
    plt.scatter(range(len(correction_error)), correction_error)

    fig2 = plt.figure()
    plt.scatter(original_full_spline[:, 0], original_full_spline[:, 1], label='original full spline points',
                color='red')
    plt.plot(arch_ct_to_ios.spline_points_fine[:, 0], arch_ct_to_ios.spline_points_fine[:, 1], '-',
             label='spline with missing teeth',
             color='green')
    plt.scatter(corrected_spline[:, 0], corrected_spline[:, 1], label='corrected spline', color='blue')
    # plt.plot(test[:,0], test[:,1], label='test')
    plt.legend()
    plt.show()

    # Update all points for drawing
    arch_ios.update_all_teeth_points(missing_tooth_flag=True)
    arch_ct_in_ios.update_all_teeth_points(missing_tooth_flag=True)
    arch_ct_to_ios.update_all_teeth_points(missing_tooth_flag=True)
    arch_ct_in_ios_curvilinear_correction.update_all_teeth_points(missing_tooth_flag=True)
    arch_ios_curvilinear_correction.update_all_teeth_points(missing_tooth_flag=True)

    print('checking rigid transformation')
    draw_registration_result_points_array(arch_ios.allpoints, arch_ct_to_ios.allpoints, np.eye(4))
    print('checking local ICP')
    draw_registration_result_points_array(arch_ios.allpoints, arch_ct_in_ios.allpoints, np.eye(4))
    print('checking curvilinear correction')
    draw_registration_result_points_array(arch_ios_curvilinear_correction.allpoints, arch_ct_to_ios.allpoints,
                                          np.eye(4))
    print('checking curvilinear correction guided points')
    draw_registration_result_points_array(arch_ct_in_ios_curvilinear_correction.allpoints, arch_ct_to_ios.allpoints,
                                          np.eye(4))
    exit()
