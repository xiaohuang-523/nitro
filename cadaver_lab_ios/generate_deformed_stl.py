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
from stl import mesh


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
    # fiducial_list_yomiplan = range(20)
    # yomi_plan_fiducial_ios = []
    # yomi_plan_fiducial_ground = []
    # for i in fiducial_list_yomiplan:
    #     yomi_plan_fiducial_ios.append(splint_fiducial_ios_transformed[i,:])
    #     yomi_plan_fiducial_ground.append(splint_fiducial_ground[i,:])
    # yomi_plan_fiducial_ios = np.asarray(yomi_plan_fiducial_ios)
    # yomi_plan_fiducial_ground = np.asarray(yomi_plan_fiducial_ground)
    # yomi_plan_fiduial_fiducialFrame = transpose_pc(yomi_plan_fiducial_ground, fiducial_frame)
    # yomi_plan_fiduial_fiducialFrame = transpose_pc(yomi_plan_fiduial_fiducialFrame, golden_transformation)



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
    # for i in arch_ios.existing_tooth_list:
    #     points_tem = arch_ios.get_tooth(i).points
    #     points_tem_transformed = transpose_pc(points_tem, trans_rigid0)
    #     points_tem_transformed_cylindrical = coordinates.convert_cylindrical(points_tem_transformed, splint_ground.spline_points_fine_cylindrical_mid_points)
    #     points_tem_corrected = sc.displacement(points_tem_transformed, points_tem_transformed_cylindrical, splint_ground.spline_points_fine_cylindrical_mid_points, displacement_splint)
    #     tooth_feature_splint_correction = fe.ToothFeature(points_tem_corrected, i, 'solidworks', 'IOS')
    #     arch_ios_splint_correction.add_tooth(i, tooth_feature_splint_correction)

    # Deform stl image and save to new stl
    your_mesh = mesh.Mesh.from_file("G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\demo_scan\\2nd_splint\\full_arch2\\5272021-demo_picture_full_arch-lowerjaw.stl")
    original_stl_points = np.copy(your_mesh.points)
    new_stl_points = np.copy(your_mesh.points)
    for j in range(3):
        points_tem = np.copy(original_stl_points[:, 3*j:3*(j+1)])
        points_tem_transformed = transpose_pc(points_tem, trans_rigid0)
        points_tem_transformed_cylindrical = coordinates.convert_cylindrical(points_tem_transformed,
                                                                             splint_ground.spline_points_fine_cylindrical_mid_points)
        points_tem_corrected = sc.displacement(points_tem_transformed, points_tem_transformed_cylindrical,
                                               splint_ground.spline_points_fine_cylindrical_mid_points,
                                               displacement_splint)
        new_stl_points[:,3*j:3*(j+1)] = points_tem_corrected

    for i in range(np.shape(new_stl_points)[0]):
        your_mesh.points[i,:] = new_stl_points[i,:]
    print('original_stl_points 0 are', original_stl_points[0,:])
    print('new_stl_points 0 are', new_stl_points[0,:])
    print('your_mesh points 0 are', your_mesh.points[0,:])
    your_mesh.save("G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\demo_scan\\2nd_splint\\full_arch2\\deformed.stl")