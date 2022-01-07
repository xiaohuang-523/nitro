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
from os import path


def transpose_pc(pc_2b_convert, transformation):
    pc_converted = []
    if pc_2b_convert.ndim == 1:
        tem = pc_2b_convert
        tem = np.insert(tem, 3, 1.)
        tem_converted = np.matmul(transformation, tem)[0:3]
        pc_converted = tem_converted
    else:
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
    #TOOTH_NUMBER = np.array([17, 24])
    MISSING_TOOTH_NUMBER = [18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]
    IGNORE_TOOTH_NUMBER = [18, 19, 20, 21, 22, 23]
    MISSING_TOOTH_NUMBER_TARGET = [31]
    NEIGHBOUR_TOOTH = [17, 24]
    NEIGHBOUR_TOOTH_TARGET = [30, 32]
    #NEIGHBOUR_TOOTH_TARGET = [30]
    TARGET_TOOTH = [31]
    ORIGINAL_NO_MISSING = []
    FXT_ALIGN_TOOTH = [17, 31]  # middle tooth in missing tooth number
    NUMBER_SPLINE_GUIDED_POINTS = 100

    # ---- Define global file paths
    RESULT_TEM_BASE = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\TRE_verification\\result\\"

    # ---- Extract features
    source_stl_base = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\TRE_verification\\stl_pc\\"
    target_dicom_base = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\TRE_verification\\dicom_pc\\"
    faro_measurement_base = "G:\\My Drive\\Project\\IntraOral Scanner Registration" \
                            "\\TRE_verification\\faro_measurement\\"
    splint_base = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\TRE_verification\\splint_geometry\\"

    stl_file_name = 'new_stl_points_tooth'
    dicom_file_name = 'corrected_dicom_points_tooth'
    faro_file_name = 'faro_measurement'
    splint_designed = 'designed_geometry.txt'
    splint_ios = 'tested_geometry.txt'

    # ---- Read splint measurement
    splint_fiducial_ground = Yomiread.read_csv(splint_base + splint_designed, 4)[:, 1:]
    splint_fiducial_ios = Yomiread.read_csv(splint_base + splint_ios, 4)[:, 1:]

    # initialize arch classes
    arch_ios = fe.FullArch(MISSING_TOOTH_NUMBER, 'IOS', 'IOS')
    arch_ct = fe.FullArch(MISSING_TOOTH_NUMBER, 'CT', 'CT')

    # arch after 1st step correction (splint correction)
    arch_ios_splint_correction = fe.FullArch(MISSING_TOOTH_NUMBER, 'IOS', 'IOS')

    # convert ct features to IOS space with rigid transformation
    arch_ct_to_ios = fe.FullArch(MISSING_TOOTH_NUMBER, 'CT', 'IOS')

    # convert IOS features to ct space with rigid transformation
    arch_ios_to_ct = fe.FullArch(MISSING_TOOTH_NUMBER, 'IOS', 'CT')

    # generate virtual landmarks using ct features with local ICP
    arch_ct_in_ios = fe.FullArch(MISSING_TOOTH_NUMBER, 'CT', 'IOS')

    # arch after curvilinear correction - original ios arch
    arch_ios_curvilinear_correction = fe.FullArch(MISSING_TOOTH_NUMBER, 'IOS', 'IOS')

    # arch after curvilinear correction - virtual ct features
    arch_ct_in_ios_curvilinear_correction = fe.FullArch(MISSING_TOOTH_NUMBER, 'CT', 'IOS')

    # Read CT and IOS full arch scan
    for i in TOOTH_NUMBER:
        print('reading CT tooth', i)
        if i in MISSING_TOOTH_NUMBER:
            print('missing tooth', i)
        else:
            new_dicom_path = target_dicom_base + dicom_file_name + np.str(i) + '.csv'
            dicom_tem = Yomiread.read_csv(new_dicom_path, 3, -1)
            tooth_dicom_tem = fe.ToothFeature(dicom_tem, i, 'CT', 'CT')
            arch_ct.add_tooth(i, tooth_dicom_tem)

        print('reading IOS tooth', i)
        if i in MISSING_TOOTH_NUMBER:
            print('Tooth '+np.str(i) + ' is covered by splint')
            if i in TARGET_TOOTH:
                print('Tooth ' + np.str(i) + ' is covered by target')
        else:
            new_stl_path = source_stl_base + stl_file_name + np.str(i) + '.csv'
            stl_tem = Yomiread.read_csv(new_stl_path, 3, -1)
            tooth_stl_tem = fe.ToothFeature(stl_tem, i, 'IOS', 'IOS')
            arch_ios.add_tooth(i, tooth_stl_tem)
        if 'stl_tem' in locals():
            del stl_tem
        if 'dicom_tem' in locals():
            del dicom_tem

    # Add target points to ios arch
    arch_ios.add_target(splint_fiducial_ios[20:,:])

    # ---- Step 1 Splint correction
    # transform ios splint to splint ground geometry (solidworks space)
    trans_init = np.eye(4)
    rigid_init_parameter0 = Yomikin.Yomi_parameters(trans_init)
    affine_rigid_part0 = affine_registration.rigid_registration(rigid_init_parameter0, splint_fiducial_ground[0:20, :], splint_fiducial_ios[0:20, :])
    trans_rigid0 = Yomikin.Yomi_Base_Matrix(affine_rigid_part0)

    # modify the orientation (rotation) for spline fitting.
    #modify_parameters = np.array([0, 0, 0, -np.pi / 2, 0, 0])
    #modify_matrix = Yomikin.Yomi_Base_Matrix(modify_parameters)
    #modify_parameters2 = np.array([0, 0, 0, 5*np.pi / 12, 0, 0])
    #modify_matrix2 = Yomikin.Yomi_Base_Matrix(modify_parameters2)

    splint_fiducial_ios_transformed = transpose_pc(splint_fiducial_ios, trans_rigid0)

    fig = plt.figure()
    plt.scatter(splint_fiducial_ground[0:20, 0], splint_fiducial_ground[0:20, 1])
    plt.scatter(splint_fiducial_ios_transformed[0:20, 0], splint_fiducial_ios_transformed[0:20, 1])
    plt.show()

    # convert true fiducial in fiducial frame
    sphere1 = splint_fiducial_ground[20, :]
    sphere2 = splint_fiducial_ground[22, :]
    sphere3 = splint_fiducial_ground[21, :]
    fiducial_frame = coordinates.generate_frame_yomi(sphere1, sphere2, sphere3)

    golden_transformation = np.array([[-0.000185716, 0.999999,-0.00122936, -12.702999999999999],
                                      [-0.999999,-0.000184434,0.00104337, 4.058],
                                      [0.00104315, 0.00122956,0.999999, -7.938],
                                      [0, 0, 0, 1]])

    # test rigid registration quality
    delta = splint_fiducial_ground[0:11,:] - splint_fiducial_ios_transformed[0:11,:]
    print('delta is', np.linalg.norm(delta, axis=1))

    # Initialize splint clasess
    fiducial_list = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
    splint_ground = fe.Splint(fiducial_list, type='Geometry', spline_base_axis='x')
    splint_ios = fe.Splint(fiducial_list, type='IOS', spline_base_axis='x')
    splint_ios_corrected = fe.Splint(fiducial_list, type='IOS', spline_base_axis='x')

    # Get the fiducials for Yomiplan registration demo
    # list is [0, 6, 10, 12, 15, 19]
    #fiducial_list_yomiplan = [0, 6, 10, 12, 15, 19]
    fiducial_list_yomiplan = range(20)
    yomi_plan_fiducial_ios = []
    yomi_plan_fiducial_ground = []
    for i in fiducial_list_yomiplan:
        yomi_plan_fiducial_ios.append(splint_fiducial_ios_transformed[i,:])
        yomi_plan_fiducial_ground.append(splint_fiducial_ground[i,:])
    yomi_plan_fiducial_ios = np.asarray(yomi_plan_fiducial_ios)
    yomi_plan_fiducial_ground = np.asarray(yomi_plan_fiducial_ground)
    yomi_plan_fiduial_fiducialFrame = transpose_pc(yomi_plan_fiducial_ground, fiducial_frame)
    yomi_plan_fiduial_fiducialFrame = transpose_pc(yomi_plan_fiduial_fiducialFrame, golden_transformation)



    for i in range(np.shape(splint_fiducial_ios)[0]):
        splint_ground.add_pyramid(i, splint_fiducial_ground[i, :])
        splint_ios.add_pyramid(i, splint_fiducial_ios_transformed[i, :])

    print('fiducial list is', splint_ground.pyramid_fiducial_list)
    print('target list is', splint_ground.pyramid_target_list)
    print('all list is', splint_ground.pyramid_number_list)

    splint_ground.update_spline(fine_flag=True)
    splint_ios.update_spline(fine_flag=True)

    # perform splint correction
    displacement_splint = np.asarray(splint_ground.spline_points_fine) - np.asarray(splint_ios.spline_points_fine)
    for i in arch_ios.existing_tooth_list:
        points_tem = arch_ios.get_tooth(i).points
        points_tem_transformed = transpose_pc(points_tem, trans_rigid0)
        points_tem_transformed_cylindrical = coordinates.convert_cylindrical(points_tem_transformed, splint_ground.spline_points_fine_cylindrical_mid_points)
        points_tem_corrected = sc.displacement(points_tem_transformed, points_tem_transformed_cylindrical, splint_ground.spline_points_fine_cylindrical_mid_points, displacement_splint)
        tooth_feature_splint_correction = fe.ToothFeature(points_tem_corrected, i, 'solidworks', 'IOS')
        arch_ios_splint_correction.add_tooth(i, tooth_feature_splint_correction)

    #target_points_transfromed = transpose_pc(splint_fiducial_ios[20:, :], trans_rigid0)
    #target_points_transfromed_cylindrical = coordinates.convert_cylindrical(target_points_transfromed, splint_ground.spline_points_fine_cylindrical_mid_points)
    #target_points_corrected = sc.displacement(target_points_transfromed, target_points_transfromed_cylindrical, splint_ground.spline_points_fine_cylindrical_mid_points, displacement_splint)
    #arch_ios_splint_correction.add_target(target_points_corrected)
    #print('original target points are', splint_fiducial_ios[20:, :])
    #print('target_points corrected are', target_points_corrected)
    #print('target_points corrected check are', arch_ios_splint_correction.target_points)

    local_ICP.do_local_registration(TRANS_INIT, THRESHOLD_ICP, RMS_LOCAL_REGISTRATION, arch_ios_splint_correction, arch_ct, DEBUG=0)
    # check local registration quality (rms of local ICP of each tooth)
    for i in arch_ios_splint_correction.existing_tooth_list:  # used for algorithm verification (count missing tooth as well)
        # for i in arch_ios.existing_tooth_list: # used for real 3 images processing
        print('tooth ' + np.str(i) + ' ICP rms is ' + np.str(arch_ios_splint_correction.get_tooth(i).ICP.inlier_rmse))

    # ---- Step 2 Arch correction
    # Transform ct features to ios space using ICP transformation
    for i in arch_ct_in_ios.existing_tooth_list:
        print('transform tooth ', i)
        points_ICP_tem = transpose_pc(arch_ct.get_tooth(i).points, arch_ct.get_tooth(i).local_ICP_transformation)
        tooth_feature_ICP_tem = fe.ToothFeature(points_ICP_tem, i, 'CT', 'IOS')
        arch_ct_in_ios.add_tooth(i, tooth_feature_ICP_tem)

    # generate virtual landmarks
    ctl_ct = []
    ctl_ios = []
    for i in NEIGHBOUR_TOOTH:
        ctl_ct.append(arch_ct.get_tooth(i).points)
        ctl_ios.append(arch_ct_in_ios.get_tooth(i).points)
    ctl_ct = combine_pc(ctl_ct)
    ctl_ios = combine_pc(ctl_ios)

    # ctl_target2 = []
    # ctl_source2 = []
    # for i in NEIGHBOUR_TOOTH_TARGET:
    #     ctl_target2.append(arch_ct.get_tooth(i).points)
    #     ctl_source2.append(arch_ct_in_ios.get_tooth(i).points)
    # ctl_source2 = combine_pc(ctl_source2)
    # ctl_target2 = combine_pc(ctl_target2)

    trans_init1 = arch_ios_splint_correction.get_tooth(17).local_ICP_transformation
    print('transformation check is', np.matmul(arch_ios_splint_correction.get_tooth(17).local_ICP_transformation, arch_ct.get_tooth(17).local_ICP_transformation))
    rigid_init_parameter1 = Yomikin.Yomi_parameters(trans_init1)
    if path.exists(RESULT_TEM_BASE + 'optimized_rigid_transformation_1st.csv'):
        trans_rigid = Yomiread.read_csv(RESULT_TEM_BASE + 'optimized_rigid_transformation_1st.csv', 4, 4)
    else:
        affine_rigid_part = affine_registration.rigid_registration(rigid_init_parameter1, ctl_ct, ctl_ios)
        trans_rigid = Yomikin.Yomi_Base_Matrix(affine_rigid_part)
        Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'optimized_rigid_transformation_1st.csv', trans_rigid,
                                   fmt='%0.8f')

    yomi_plan_fiducial_ios_ct = transpose_pc(yomi_plan_fiducial_ios, trans_rigid)
    yomi_plan_fiducial_ct = transpose_pc(yomi_plan_fiducial_ground, trans_rigid)
    print('yomi_plan_fiducial difference is', yomi_plan_fiducial_ct-yomi_plan_fiducial_ios_ct)

    #fiducial_test = transpose_pc(yomi_plan_fiducial_ct, np.linalg.inv(trans_rigid))
    fiducial_test = yomi_plan_fiducial_ground
    fiducial_test = transpose_pc(fiducial_test, fiducial_frame)
    fiducial_test = transpose_pc(fiducial_test, golden_transformation)
    print('delta is', fiducial_test - yomi_plan_fiduial_fiducialFrame)

    mtx = np.matmul(golden_transformation, fiducial_frame)
    mtx = np.matmul(mtx, np.linalg.inv(trans_rigid))

    check_pc = transpose_pc(yomi_plan_fiducial_ct, mtx)

    all_check_pc = np.vstack((check_pc, yomi_plan_fiduial_fiducialFrame))
    plot_3d_pc(all_check_pc)

    # check ct space
    print('check ct space')
    #arch_ct.update_all_teeth_points(missing_tooth_flag=True)
    arch_ct.update_all_teeth_points()
    splint_fiducial_ground_ct = transpose_pc(splint_fiducial_ground, trans_rigid)
    check_pc2 = np.vstack((arch_ct.allpoints, splint_fiducial_ground_ct))
    plot_3d_pc(check_pc2)

    # check fiducial space
    print('check fiducial space')
    allpoints_fiducial_space = transpose_pc(arch_ct.allpoints, mtx)
    check_pc3 = np.vstack((allpoints_fiducial_space, yomi_plan_fiduial_fiducialFrame))
    plot_3d_pc(check_pc3)

# ------ the following are defined for full_arch application ------ #
    #landmark_ct = yomi_plan_fiducial_ct[12,:]
    #implant_ct = np.array([[(800 - 285.9) * 0.2, 264.834 * 0.2, (130 + 304-200)*0.2],
    #                       [(800 - 286.881) * 0.2, 265.624 * 0.2, (130 + 304-210)*0.2]])
    #landmark_ct_2 = np.array([(800-230)*0.2, 390*0.2, (130+304-283)*0.2])
    #landmark_ct_3 = np.array([(800-334.926)*0.2, 496.589*0.2, (130+304-279)*0.2])
    #landmark_ct_4 = np.array([(800-468.541)*0.2, 502.463*0.2, (130+304-273)*0.2])
    #landmark_ct_tooth = np.array([(800-257)*0.2, 228*0.2, (130+304-169)*0.2])


# ------ for drill fixture ------ #
    # landmark on splint
    landmark_ct = yomi_plan_fiducial_ct[12, :]

    # 7 implants
    implant_x = np.array([0, 0, -1])
    implant_y = np.array([1, 0, 0])
    implant_z = np.cross(implant_x, implant_y)
    implant_transformation_ct = np.eye(4)
    implant_transformation_ct[0:3,0] = implant_x
    implant_transformation_ct[0:3,1] = implant_y
    implant_transformation_ct[0:3,2] = implant_z

    # implant 1
    implant1_ct_o = np.array([(800 - (302+335)/2)*0.2, (281 + 314)/2 * 0.2, (50+304-91)*0.2 - 4.25])
    implant1_transformation_ct = implant_transformation_ct
    implant1_transformation_ct[0:3,3] = implant1_ct_o
    implant1_transformation_fiducial = np.matmul(mtx, implant1_transformation_ct)
    # implant 2
    implant2_ct_o = np.array([(800 - (315+345)/2)*0.2, (317 + 352)/2 * 0.2, (50+304-91)*0.2- 4.25])
    implant2_transformation_ct = implant_transformation_ct
    implant2_transformation_ct[0:3,3] = implant2_ct_o
    implant2_transformation_fiducial = np.matmul(mtx, implant2_transformation_ct)
    # implant 3
    implant3_ct_o = np.array([(800 - (329+364)/2)*0.2, (353 + 389)/2 * 0.2, (50+304-91)*0.2- 4.25])
    implant3_transformation_ct = implant_transformation_ct
    implant3_transformation_ct[0:3,3] = implant3_ct_o
    implant3_transformation_fiducial = np.matmul(mtx, implant3_transformation_ct)
    # implant 4
    implant4_ct_o = np.array([(800 - (345.5+378.5)/2)*0.2, (390 + 426)/2 * 0.2, (50+304-91)*0.2- 4.25])
    implant4_transformation_ct = implant_transformation_ct
    implant4_transformation_ct[0:3,3] = implant4_ct_o
    implant4_transformation_fiducial = np.matmul(mtx, implant4_transformation_ct)
    # implant 5
    implant5_ct_o = np.array([(800 - (362.5+395.5)/2)*0.2, (427 + 462)/2 * 0.2, (50+304-91)*0.2- 4.25])
    implant5_transformation_ct = implant_transformation_ct
    implant5_transformation_ct[0:3,3] = implant5_ct_o
    implant5_transformation_fiducial = np.matmul(mtx, implant5_transformation_ct)
    # implant 6
    implant6_ct_o = np.array([(800 - (384.5+417.5)/2)*0.2, (462.5 + 497.5)/2 * 0.2, (50+304-91)*0.2- 4.25])
    implant6_transformation_ct = implant_transformation_ct
    implant6_transformation_ct[0:3,3] = implant6_ct_o
    implant6_transformation_fiducial = np.matmul(mtx, implant6_transformation_ct)
    # implant 7
    implant7_ct_o = np.array([(800 - (416+449.5)/2)*0.2, (488 + 523.5)/2 * 0.2, (50+304-91)*0.2- 4.25])
    implant7_transformation_ct = implant_transformation_ct
    implant7_transformation_ct[0:3,3] = implant7_ct_o
    implant7_transformation_fiducial = np.matmul(mtx, implant7_transformation_ct)

    landmark_ct_2 = np.array([(800-319)*0.2, 366*0.2, (50+304-91)*0.2])
    landmark_ct_3 = np.array([(800-370)*0.2, 385*0.2, (50+304-91)*0.2])
    landmark_ct_4 = np.array([(800-373.5)*0.2, 472.5*0.2, (50+304-91)*0.2])

    landmark_fiducial = transpose_pc(landmark_ct, mtx)
    landmark_fiducial2 = transpose_pc(landmark_ct_2, mtx)
    landmark_fiducial3 = transpose_pc(landmark_ct_3, mtx)
    landmark_fiducial4 = transpose_pc(landmark_ct_4, mtx)

    #297, 625

    point0 = np.array([(800-625)*0.2, 297*0.2, (50+304-61) * 0.2])
    point = np.vstack((point0, point0 + 2*implant_y))
    point = np.vstack((point, point0 + 4 * implant_y))
    point = np.vstack((point, point0 + 6 * implant_y))
    point = np.vstack((point, point0 + 8 * implant_y))
    point = np.vstack((point, point0 + 10 * implant_y))
    point = np.vstack((point, point0 + 12 * implant_y))
    point = np.vstack((point, point0 + 14 * implant_y))
    point = np.vstack((point, point0 + 16 * implant_y))

    print('check landmark and implant in ct space')
    check_pc4 = np.vstack((arch_ct.allpoints, landmark_ct))
    check_pc4 = np.vstack((check_pc4, landmark_ct_2))
    check_pc4 = np.vstack((check_pc4, landmark_ct_3))
    check_pc4 = np.vstack((check_pc4, landmark_ct_4))
    check_pc4 = np.vstack((check_pc4, point))
    plot_3d_pc(check_pc4)

    print('check landmark and implant in fiducial space')
    check_pc5 = np.vstack((allpoints_fiducial_space, landmark_fiducial))
    check_pc5 = np.vstack((check_pc5, landmark_fiducial2))
    check_pc5 = np.vstack((check_pc5, landmark_fiducial3))
    check_pc5 = np.vstack((check_pc5, landmark_fiducial4))
    plot_3d_pc(check_pc5)

    # convert python ct point coordinate to Yomiplan
    # Y is flipped with y_data = (slice_properties[4] - surface_idx[i][0]) * slice_properties[1][0]
    # Z is flipped with z_data = (slice_properties[2] - z) * slice_properties[0]
    # In the current dicom, slice_properties are
    # [slice_thickness, slice_pixel_spacing, n, slice_rows, slice_columns, slice_columns, slope, intercept]
    # ["0.2", [0.200, 0.200], 304, 800, 800, "1.0", "0.0"]

    arch_ct.update_spline()
    check_centroids = np.asarray(arch_ct.spline_points)
    check_centroids_ori = np.copy(check_centroids)

    #
    # Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'fiducials_in_CT_space.csv', yomi_plan_fiducial_ct_test,
    #                            fmt='%0.8f', delim=' ')
    # Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'fiducials_in_fiducial_space.csv', yomi_plan_fiduial_fiducialFrame,
    #                            fmt='%0.8f', delim=' ')
    #Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'spline_points_in_space.csv', check_centroids,
    #                            fmt='%0.8f', delim=' ')

    #Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'fiducials_in_CT_space.csv', test_ct_frame_fiducials,
    #                           fmt='%0.8f')
    #Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'fiducials_in_fiducial_space.csv', test_fiducial_frame_fiducials,
    #                           fmt='%0.8f')

    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'landmarkss_in_fiducial_space.csv', landmark_fiducial,
                               fmt='%0.8f')
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'landmarkss2_in_fiducial_space.csv', landmark_fiducial2,
                               fmt='%0.8f')
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'landmarkss3_in_fiducial_space.csv', landmark_fiducial3,
                               fmt='%0.8f')
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'landmarkss4_in_fiducial_space.csv', landmark_fiducial4,
                               fmt='%0.8f')
    #Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'landmarkss_tooth_in_fiducial_space.csv', landmark_fiducial_tooth,
    #                           fmt='%0.8f')

    #Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'implant_in_fiducial_space.csv', implant_fiducial,
    #                           fmt='%0.8f')
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'implant_transforamtion1_in_fiducial_space.csv', implant1_transformation_fiducial,
                               fmt='%0.8f')
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'implant_transforamtion2_in_fiducial_space.csv', implant2_transformation_fiducial,
                               fmt='%0.8f')
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'implant_transforamtion3_in_fiducial_space.csv', implant3_transformation_fiducial,
                               fmt='%0.8f')
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'implant_transforamtion4_in_fiducial_space.csv', implant4_transformation_fiducial,
                               fmt='%0.8f')
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'implant_transforamtion5_in_fiducial_space.csv', implant5_transformation_fiducial,
                               fmt='%0.8f')
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'implant_transforamtion6_in_fiducial_space.csv', implant6_transformation_fiducial,
                               fmt='%0.8f')
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'implant_transforamtion7_in_fiducial_space.csv', implant7_transformation_fiducial,
                               fmt='%0.8f')


    draw_registration_result_points_array(ctl_ios, ctl_ct, trans_rigid)

    #point1 = splint_ground.pyramid_fiducial_points
    #arch_ct.update_all_teeth_points(missing_tooth_flag=True)

    #pc_new = np.vstack((arch_ct.allpoints, yomi_plan_fiducial_ct))
    #pc_new = np.vstack((pc_new, implant_ct))
    #plot_3d_pc(pc_new)


    exit()



    print('finish fiducial transformation, stop program')
    exit()

    trans_init2 = arch_ct.get_tooth(29).local_ICP_transformation
    rigid_init_parameter2 = Yomikin.Yomi_parameters(trans_init2)
    if path.exists(RESULT_TEM_BASE + 'optimized_rigid_transformation_2nd.csv'):
        trans_rigid_target = Yomiread.read_csv(RESULT_TEM_BASE + 'optimized_rigid_transformation_2nd.csv', 4, 4)
    else:
        affine_rigid_part_target = affine_registration.rigid_registration(rigid_init_parameter2, ctl_source2, ctl_target2)
        trans_rigid_target = Yomikin.Yomi_Base_Matrix(affine_rigid_part_target)
        Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + 'optimized_rigid_transformation_2nd.csv', trans_rigid_target,
                                   fmt='%0.8f')

    # Transform CT points to the splint corrected IOS image
    # The results will be used to guide the deformation.
    for i in arch_ct.tooth_list:
        print('transform tooth ', i)
        points_rigid_tem = transpose_pc(arch_ct.get_tooth(i).points, trans_rigid)
        tooth_feature_rigid_tem = fe.ToothFeature(points_rigid_tem, i, 'IOS', 'CT')
        arch_ct_to_ios.add_tooth(i, tooth_feature_rigid_tem)
        del points_rigid_tem, tooth_feature_rigid_tem

    # Add virtual spline points (the splines points of missing teeth from ct_to_ios)
    # This can be done since the 'splint' has been corrected and matched to ct_to_ios image.
    # In theory, this part should contain the same information.
    for i in MISSING_TOOTH_NUMBER:
        if i not in TARGET_TOOTH:
            virtual_point_tem = transpose_pc(arch_ct.get_tooth(i).centroid, trans_rigid)
            arch_ct_in_ios.add_virtual_spline_point(i, virtual_point_tem)

            points_virtual_tem = transpose_pc(arch_ct.get_tooth(i).points, trans_rigid)
            tooth_feature_virtual_tem = fe.ToothFeature(points_virtual_tem, i, 'IOS', 'CT')
            arch_ct_in_ios.add_tooth(i, tooth_feature_virtual_tem)
        else:
            virtual_point_tem = transpose_pc(arch_ct.get_tooth(i).centroid, trans_rigid_target)
            arch_ct_in_ios.add_virtual_spline_point(i, virtual_point_tem)

            points_virtual_tem = transpose_pc(arch_ct.get_tooth(i).points, trans_rigid_target)
            tooth_feature_virtual_tem = fe.ToothFeature(points_virtual_tem, i, 'IOS', 'CT')
            arch_ct_in_ios.add_tooth(i, tooth_feature_virtual_tem)
        del virtual_point_tem

    # Update spline points

    arch_ct_in_ios.update_spline()
    arch_ct_in_ios.update_missing_spline()
    arch_ct_in_ios.update_virtual_guided_spline(fine_flag=True)

    arch_ct_to_ios.update_spline()
    arch_ct_to_ios.update_missing_spline()
    arch_ct_to_ios.update_guided_spline(fine_flag=True)

    arch_ct_in_ios.update_ignore_boundary(ignore_tooth_list=IGNORE_TOOTH_NUMBER)
    arch_ct_to_ios.update_ignore_boundary(ignore_tooth_list=IGNORE_TOOTH_NUMBER)
    print('arch_ct_ios_ignore boundary is', arch_ct_in_ios.ignore_boundary)

    displacement = arch_ct_to_ios.spline_guided_points_fine - arch_ct_in_ios.spline_virtual_guided_points_fine

    for i in arch_ct_to_ios.tooth_list:  # for tooth in target
        candidate_tooth = arch_ct_in_ios.get_tooth(i).points
        candidate_tooth_cylindrical = coordinates.convert_cylindrical(candidate_tooth,
                                                                      arch_ct_in_ios.spline_points_cylindrical_center)

        corrected_tooth = sc.displacement_partial_version2(candidate_tooth, candidate_tooth_cylindrical,
                                                           arch_ct_in_ios.spline_virtual_guided_points_fine_cylindrical_mid_points,
                                                           arch_ct_in_ios.ignore_boundary,
                                                           displacement)

        corrected_tooth_feature = fe.ToothFeature(corrected_tooth, i, 'CT', 'IOS')
        arch_ct_in_ios_curvilinear_correction.add_tooth(i, corrected_tooth_feature)
        del candidate_tooth

    # ---- Step 3 get final corrected target points
    target_points_cylindrical = coordinates.convert_cylindrical(np.copy(arch_ios_splint_correction.target_points),
                                                                            arch_ct_in_ios.spline_points_cylindrical_center)

    #target_points_corrected_final = sc.displacement_partial_version2(np.copy(arch_ios_splint_correction.target_points), target_points_cylindrical,
    #                                                       arch_ct_in_ios.spline_virtual_guided_points_fine_cylindrical_mid_points,
    #                                                       arch_ct_in_ios.ignore_boundary,
    #                                                       displacement)

    target_points_corrected_final = sc.displacement_partial_version2_target(np.copy(arch_ios_splint_correction.target_points), target_points_cylindrical,
                                                           arch_ct_in_ios.spline_virtual_guided_points_fine_cylindrical_mid_points,
                                                           arch_ct_in_ios.ignore_boundary,
                                                           displacement)

    print('original target points are', splint_fiducial_ios[20:, :])
    print('target_points corrected are', target_points_corrected)
    print('target_points corrected check are', arch_ios_splint_correction.target_points)

    print('target_points_corrected final are', target_points_corrected_final)
    frame_points = splint_fiducial_ground_transformed[20:,:]
    print('splint designed frame are', frame_points)

    frame_points = transpose_pc(frame_points, modify_matrix2)
    print('splint designed frame corrected are', frame_points)

    result_file_frame_points = 'frame_points.txt'
    result_file_target_points = 'target_points.txt'
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE+result_file_frame_points,frame_points, fmt='%.6f')
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + result_file_target_points, target_points_corrected_final, fmt='%.6f')

    #arch_ios_curvilinear_correction.update_spline()
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
    plt.savefig(RESULT_TEM_BASE + "matching_error_missing_teeth" + np.str(MISSING_TOOTH_NUMBER[0]) + '_' + np.str(MISSING_TOOTH_NUMBER[1]) + '_' + np.str(MISSING_TOOTH_NUMBER[2]))
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + "matching_error_missing_teeth" + np.str(MISSING_TOOTH_NUMBER[0]) + '_' + np.str(MISSING_TOOTH_NUMBER[1]) + '_' + np.str(MISSING_TOOTH_NUMBER[2]) + '.txt', correction_error)

    fig2 = plt.figure()
    plt.scatter(original_full_spline[:, 0], original_full_spline[:, 1], label='Spline Landmarks',
                color='red')
    plt.plot(arch_ct_to_ios.spline_guided_points_fine[:, 0], arch_ct_to_ios.spline_guided_points_fine[:, 1], '-',
             label='Fitted Spline',
             color='green')
   # plt.scatter(corrected_spline[:, 0], corrected_spline[:, 1], label='corrected spline', color='blue')
    # plt.plot(test[:,0], test[:,1], label='test')

    plt.title('Spline fitting along full arch (2D plot)')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.show()

    # Update all points for drawing
    arch_ios.update_all_teeth_points(missing_tooth_flag=False)
    arch_ct_in_ios.update_all_teeth_points(missing_tooth_flag=True)
    arch_ct_to_ios.update_all_teeth_points(missing_tooth_flag=True)
    arch_ct_in_ios_curvilinear_correction.update_all_teeth_points(missing_tooth_flag=True)
    #arch_ios_curvilinear_correction.update_all_teeth_points(missing_tooth_flag=True)

    print('checking rigid transformation')
    draw_registration_result_points_array(arch_ct_in_ios.allpoints, arch_ct_to_ios.allpoints, np.eye(4))
    #print('checking local ICP')
    #draw_registration_result_points_array(arch_ios.allpoints, arch_ct_in_ios.allpoints, np.eye(4))
    #print('checking curvilinear correction')
    #draw_registration_result_points_array(arch_ios_curvilinear_correction.allpoints, arch_ct_to_ios.allpoints,
    #                                      np.eye(4))
    print('checking curvilinear correction guided points')
    draw_registration_result_points_array(arch_ct_in_ios_curvilinear_correction.allpoints, arch_ct_to_ios.allpoints,
                                          np.eye(4))
    exit()
