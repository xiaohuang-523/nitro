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


def transpose_pc(pc_2b_convert, transformation):
    pc_converted = []
    for point in pc_2b_convert:
        tem = np.insert(point, 3, 1.)
        tem_converted = np.matmul(transformation, tem)[0:3]
        pc_converted.append(tem_converted)
    return np.asarray(pc_converted)


def combine_pc(list):
    combined_pc = list[0]
    for i in range(len(list)-1):
        #combined_pc = np.vstack((combined_pc, list[i+1]))
        combined_pc = np.concatenate((combined_pc,list[i+1]))
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
        r.append(np.sqrt(dx**2 + dy**2))
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
    MISSING_TOOTH_NUMBER = [28, 29, 30]
    ORIGINAL_NO_MISSING = []
    #MISSING_TOOTH_NUMBER = []
    NEIGHBOUR_TOOTH = [31]
    NUMBER_SPLINE_GUIDED_POINTS = 100

    # ---- Define global file paths
    RESULT_TEM_BASE = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\Point Cloud and Registration TEMP\\"
    
    # ---- Extract features
    source_stl_base = 'G:\My Drive\Project\IntraOral Scanner Registration\STL_pc\\'
    target_dicom_base = 'G:\My Drive\Project\IntraOral Scanner Registration\Point Cloud and Registration Raw Data\DICOM_pc\\'
    stl_file_name = 'new_stl_points_tooth'
    dicom_file_name = 'new_dicom_points_tooth'

    arch_ct_original = fe.FullArch(ORIGINAL_NO_MISSING, 'CT', 'CT')
    arch_ios = fe.FullArch(MISSING_TOOTH_NUMBER, 'IOS', 'IOS')
    arch_ct = fe.FullArch(MISSING_TOOTH_NUMBER, 'CT', 'CT')
    # convert ct features to IOS space with rigid transformation
    arch_ct_to_ios = fe.FullArch(MISSING_TOOTH_NUMBER, 'CT', 'IOS')
    # generate virtual landmarks using ct features with local ICP
    arch_ct_in_ios = fe.FullArch(MISSING_TOOTH_NUMBER, 'CT', 'IOS')
    # arch after curvilinear correction - original ios arch
    arch_ios_curvilinear_correction = fe.FullArch(MISSING_TOOTH_NUMBER, 'IOS', 'IOS')
    # arch after curvilinear correction - virtual ct features
    arch_ct_in_ios_curvilinear_correction = fe.FullArch(MISSING_TOOTH_NUMBER, 'CT', 'IOS')

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
        del stl_tem, dicom_tem

    # ---- Perform Local ICP
    local_ICP.do_local_registration(TRANS_INIT,THRESHOLD_ICP,RMS_LOCAL_REGISTRATION, arch_ios, arch_ct)
    # check local registration quality (rms of local ICP of each tooth)
    for i in arch_ios.tooth_list: # used for algorithm verification (count missing tooth as well)
    #for i in arch_ios.existing_tooth_list: # used for real 3 images processing
        print('tooth ' + np.str(i) + ' ICP rms is ' + np.str(arch_ios.get_tooth(i).ICP.inlier_rmse))

    # Update spline points
    arch_ios.update_spline()
    arch_ct.update_spline()
    arch_ct_original.update_spline()

    # ---- Perform initial alignment
    trans_init = arch_ct.get_tooth(NEIGHBOUR_TOOTH).local_ICP_transformation
    # Transform ct features to ios space
    for i in arch_ct.tooth_list:
        print('transform tooth ', i)
        points_rigid_tem = transpose_pc(arch_ct.get_tooth(i).points, trans_init)
        points_virtual_tem = transpose_pc(arch_ct.get_tooth(i).points, arch_ct.get_tooth(i).local_ICP_transformation)
        tooth_feature_rigid_tem = fe.ToothFeature(points_rigid_tem, i, 'CT', 'IOS')
        tooth_feature_virtual_tem = fe.ToothFeature(points_virtual_tem, i, 'CT', 'IOS')
        arch_ct_to_ios.add_tooth(i, tooth_feature_rigid_tem)
        arch_ct_in_ios.add_tooth(i, tooth_feature_virtual_tem)
        del points_rigid_tem, tooth_feature_rigid_tem, points_virtual_tem, tooth_feature_virtual_tem

    # Update spline points
    arch_ct_in_ios.update_spline(fine_flag=True)
    arch_ct_to_ios.update_spline(fine_flag=True)

    print('displacement check', arch_ct_to_ios.spline_points - arch_ct_in_ios.spline_points)
    print('original spline is', arch_ct_in_ios.spline_points)
    print('target spline is', arch_ct_to_ios.spline_points)
    displacement = arch_ct_to_ios.spline_points_fine - arch_ct_in_ios.spline_points_fine
    corrected_spline = sc.displacement(arch_ct_in_ios.spline_points, arch_ct_in_ios.spline_points_cylindrical, arch_ct_in_ios.spline_points_fine_cylindrical_mid_points, displacement)
    print('corrected spline is', corrected_spline)

    for i in arch_ct_to_ios.tooth_list:
        candidate_tooth = arch_ct_in_ios.get_tooth(i).points
        candidate_tooth_cylindrical = coordinates.convert_cylindrical(candidate_tooth, arch_ct_in_ios.spline_points_cylindrical_center)
        corrected_tooth = sc.displacement(candidate_tooth, candidate_tooth_cylindrical,arch_ct_in_ios.spline_points_fine_cylindrical_mid_points, displacement)
        corrected_tooth_feature = fe.ToothFeature(corrected_tooth, i, 'CT', 'IOS')
        arch_ct_in_ios_curvilinear_correction.add_tooth(i, corrected_tooth_feature)

        candidate2_tooth = arch_ios.get_tooth(i).points
        candidate2_tooth_cylindrical = coordinates.convert_cylindrical(candidate2_tooth, arch_ct_in_ios.spline_points_cylindrical_center)
        corrected2_tooth = sc.displacement(candidate2_tooth, candidate2_tooth_cylindrical,arch_ct_in_ios.spline_points_fine_cylindrical_mid_points, displacement)
        corrected2_tooth_feature = fe.ToothFeature(corrected2_tooth, i, 'IOS', 'IOS')
        arch_ios_curvilinear_correction.add_tooth(i, corrected2_tooth_feature)

        del candidate_tooth, candidate2_tooth

    arch_ios_curvilinear_correction.update_spline()
    arch_ct_in_ios_curvilinear_correction.update_spline()

    correction_error = []
    corrected_spline = []   # spline after curvilinear correction
    correction_spline = []  # spline before curvilinear correction
    original_full_spline = []    # spline of arch_ct_to_ios
    for i in arch_ct_to_ios.tooth_list:
        error = arch_ct_to_ios.get_spline_points(i) - arch_ct_in_ios_curvilinear_correction.get_spline_points(i)
        correction_error.append(np.linalg.norm(error))
        original_full_spline.append(arch_ct_to_ios.get_tooth(i).centroid)
        #original_spline.append(arch_ct_to_ios.get_spline_points(i))
        corrected_spline.append(arch_ct_in_ios_curvilinear_correction.get_spline_points(i))
        #correction_spline.append(arch_ct_in_ios.get_spline_points(i))
    corrected_spline = np.asarray(corrected_spline)
    #original_spline = np.asarray(original_spline)
    #correction_spline = np.asarray(correction_spline)

    original_full_spline = np.asarray(original_full_spline)
    fig1 = plt.figure()
    plt.scatter(range(len(correction_error)), correction_error)


    fig2 = plt.figure()
    plt.scatter(original_full_spline[:, 0], original_full_spline[:, 1], label='original full spline points',
                color='red')
    plt.plot(arch_ct_to_ios.spline_points_fine[:, 0], arch_ct_to_ios.spline_points_fine[:, 1], '-', label='spline with missing teeth',
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
    draw_registration_result_points_array(arch_ios_curvilinear_correction.allpoints, arch_ct_to_ios.allpoints, np.eye(4))
    print('checking curvilinear correction guided points')
    draw_registration_result_points_array(arch_ct_in_ios_curvilinear_correction.allpoints, arch_ct_to_ios.allpoints, np.eye(4))
    exit()


    corrected_spline = np.asarray(corrected_spline)
    #original_spline = np.asarray(original_spline)
    correction_spline = np.asarray(correction_spline)


    #fine_spline = sc.fit_spline_and_split_2(arch_ct_in_ios.spline_points)
    #print('shape is', np.shape(arch_ct_in_ios.spline_points_fine))


    fig1 = plt.figure()
    plt.scatter(arch_ct_to_ios.spline_points[:,0], arch_ct_to_ios.spline_points[:,1], label = 'original spline points', color='red')
    plt.scatter(corrected_spline[:,0], corrected_spline[:,1], label = 'corrected_spline', color='green')
    plt.plot(arch_ct_to_ios.spline_points_fine[:, 0], arch_ct_to_ios.spline_points_fine[:, 1], '-', label='fine_spline', color='blue')
    #plt.plot(corrected_guide_points[:, 0], corrected_guide_points[:, 1], label='corrected guided_points', color='blue')
    #plt.plot(test[:,0], test[:,1], label='test')
    plt.legend()

    plt.figure()
    plt.scatter(range(len(arch_ct_in_ios.spline_points_fine_cylindrical_mid_points)), arch_ct_in_ios.spline_points_fine_cylindrical_mid_points)

    plt.show()
    exit()

    # # Perform 2D spline fitting and generate guide points
    # guide_points_ct_in_ios, guide_points_theta_boundary_0 = sc.fit_spline_and_split(arch_ct_in_ios.spline_points,
    #                                                 arch_ct_in_ios.spline_points_cylindrical_center,
    #                                                 NUMBER_SPLINE_GUIDED_POINTS)
    # guide_points_ct_to_ios, guide_points_theta_boundary_2 = sc.fit_spline_and_split(arch_ct_to_ios.spline_points,
    #                                                arch_ct_to_ios.spline_points_cylindrical_center,
    #                                                 NUMBER_SPLINE_GUIDED_POINTS)
    # displacement = guide_points_ct_to_ios - guide_points_ct_in_ios  # move ct_in_ios points to ct_to_ios
    #
    #
    # test = guide_points_ct_in_ios + displacement
    # guide_points_theta_boundary_1 = guide_points_theta_boundary_0 #+ np.pi/(2*NUMBER_SPLINE_GUIDED_POINTS)

    # Perform 3D spline fitting and generate guide points
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + "spline_target_points.csv", arch_ct_to_ios.spline_points)
    Yomiwrite.write_csv_matrix(RESULT_TEM_BASE + "spline_rigid_points.csv", arch_ct_in_ios.spline_points)
    control_list = arch_ct_in_ios.existing_tooth_list
    # # step2 generate mesh grid points
    breaks_file_rigid = RESULT_TEM_BASE + 'spline_rigid_breaks.csv'
    coe_file_rigid = RESULT_TEM_BASE + 'spline_rigid_coefficients.csv'
    breaks_rigid = Yomiread.read_csv(breaks_file_rigid, len(control_list))[0]
    coe_rigid = Yomiread.read_csv(coe_file_rigid, 4)
    fine_rigid = i3d.pp_matlab(breaks_rigid, coe_rigid, 100)
    fine_ct_in_ios = fine_rigid
    #
    breaks_file_target = RESULT_TEM_BASE + 'spline_target_breaks.csv'
    coe_file_target = RESULT_TEM_BASE + 'spline_target_coefficients.csv'
    breaks_target = Yomiread.read_csv(breaks_file_target, len(control_list))[0]
    coe_target = Yomiread.read_csv(coe_file_target, 4)
    fine_ct_to_ios = i3d.pp_matlab(breaks_target, coe_target, 100)
    #
    # # step3 perform curvilinear correction with only displacement
    displacement_x = fine_ct_to_ios[0] - fine_ct_in_ios[0]
    displacement_y = fine_ct_to_ios[1] - fine_ct_in_ios[1]
    displacement_z = fine_ct_to_ios[2] - fine_ct_in_ios[2]
    displacement = [displacement_x, displacement_y, displacement_z]
    center_t, boundary_t = sp_grid.check_curvilinear_boundary(fine_ct_to_ios)
    center_r_in_t, boundary_r_in_t = sp_grid.check_curvilinear_boundary(fine_ct_in_ios)
    #
    spline_ct_in_ios = arch_ct_in_ios.spline_points
    spline_ct_in_ios_cylindrical = sp_grid.convert_cylindrical(spline_ct_in_ios, center_r_in_t)
    spline_ct_in_ios_moved = sp_grid.displacement(spline_ct_in_ios, spline_ct_in_ios_cylindrical, boundary_r_in_t, displacement)
    #
    fine_ct_in_ios_point = np.array(np.asarray(fine_ct_in_ios).transpose())
    print('fine_rigid_point is', np.shape(fine_ct_in_ios_point))
    fine_ct_in_ios_point_cylindrical = sp_grid.convert_cylindrical(fine_ct_in_ios_point, center_r_in_t)
    fine_ct_in_ios_point_moved = sp_grid.displacement(fine_ct_in_ios_point, fine_ct_in_ios_point_cylindrical, boundary_r_in_t, displacement)
    print('fine_rigid_point_moved is', np.shape(fine_ct_in_ios_point_moved))

    correction_error = []
    corrected_spline = []   # spline after curvilinear correction
    correction_spline = []  # spline before curvilinear correction
    original_spline = []    # spline of arch_ct_to_ios
    for i in arch_ct_to_ios.tooth_list:
        #error = arch_ct_to_ios.get_spline_points(i) - arch_ct_in_ios_curvilinear_correction.get_spline_points(i)
        #correction_error.append(np.linalg.norm(error))
        original_spline.append(arch_ct_to_ios.get_spline_points(i))
        #corrected_spline.append(arch_ct_in_ios_curvilinear_correction.get_spline_points(i))
        correction_spline.append(arch_ct_in_ios.get_spline_points(i))
    corrected_spline = np.asarray(corrected_spline)
    original_spline = np.asarray(original_spline)
    correction_spline = np.asarray(correction_spline)

    correction_spline_cylindrical = sp_grid.convert_cylindrical(correction_spline, center_r_in_t)
    correction_spline_moved = sp_grid.displacement(correction_spline, correction_spline_cylindrical, boundary_r_in_t, displacement)


    fig1 = plt.figure()
    plt.scatter(correction_spline_moved[:,0], correction_spline_moved[:,1], label = 'correction_spline_moved', color='red')
    #plt.scatter(correction_spline[:,0], correction_spline[:,1], label = 'correction_spline', color='green')
    plt.scatter(original_spline[:, 0], original_spline[:, 1], label='original_spline', color='blue')
    #plt.plot(corrected_guide_points[:, 0], corrected_guide_points[:, 1], label='corrected guided_points', color='blue')
    #plt.plot(test[:,0], test[:,1], label='test')
    plt.legend()
    plt.show()
    exit()

    # Perform curvilinear correction
    # Move ct_in_ios (virtual points) to ct_to_ios (rigid points)

    for i in arch_ct_in_ios.tooth_list:
        print('move tooth ', i)
        virtual_ct_points = arch_ct_in_ios.get_tooth(i).points
        virtual_ct_points_cylindrical = convert_cylindrical(virtual_ct_points, arch_ct_in_ios.spline_points_cylindrical_center)
        virtual_ct_points_moved = sc.displacement(virtual_ct_points, virtual_ct_points_cylindrical, guide_points_theta_boundary_1, displacement)
        virtual_feature = fe.ToothFeature(virtual_ct_points_moved, i, 'CT', 'IOS')
        arch_ct_in_ios_curvilinear_correction.add_tooth(i, virtual_feature)

        original_ios_points = arch_ios.get_tooth(i).points
        original_ios_points_cylindrical = convert_cylindrical(original_ios_points, arch_ct_in_ios.spline_points_cylindrical_center)
        original_ios_points_moved = sc.displacement(original_ios_points, original_ios_points_cylindrical, guide_points_theta_boundary_1, displacement)
        ios_feature = fe.ToothFeature(original_ios_points_moved, i, 'IOS', 'IOS')
        arch_ios_curvilinear_correction.add_tooth(i, ios_feature)

        del virtual_feature, ios_feature

    arch_ct_in_ios_curvilinear_correction.update_spline()
    arch_ios_curvilinear_correction.update_spline()

    # Check curvilinear correction accuracy on centroid location
    arch_ct_in_ios_curvilinear_correction.update_missing_spline()
    arch_ct_to_ios.update_missing_spline()

    correction_error = []
    corrected_spline = []   # spline after curvilinear correction
    correction_spline = []  # spline before curvilinear correction
    original_spline = []    # spline of arch_ct_to_ios
    for i in arch_ct_to_ios.tooth_list:
        error = arch_ct_to_ios.get_spline_points(i) - arch_ct_in_ios_curvilinear_correction.get_spline_points(i)
        correction_error.append(np.linalg.norm(error))
        original_spline.append(arch_ct_to_ios.get_spline_points(i))
        corrected_spline.append(arch_ct_in_ios_curvilinear_correction.get_spline_points(i))
        correction_spline.append(arch_ct_in_ios.get_spline_points(i))
    corrected_spline = np.asarray(corrected_spline)
    original_spline = np.asarray(original_spline)
    correction_spline = np.asarray(correction_spline)

    correction_spline_cylindrical = convert_cylindrical(correction_spline, arch_ct_in_ios.spline_points_cylindrical_center)
    correction_spline_moved = sc.displacement(correction_spline, correction_spline_cylindrical, guide_points_theta_boundary_1, displacement)

    guide_points_ct_in_ios_cylindrical = convert_cylindrical(guide_points_ct_in_ios, arch_ct_in_ios.spline_points_cylindrical_center)
    corrected_guide_points = sc.displacement(guide_points_ct_in_ios, guide_points_ct_in_ios_cylindrical, guide_points_theta_boundary_1, displacement)
    # correction_spline.append(spline_point_moved)
    driving_error = correction_spline - original_spline
    driving_error = np.linalg.norm(driving_error, axis=1)

    #verify_error = corrected_guide_points - original_spline
    #verify_error = np.linalg.norm(verify_error, axis=1)

    fig1 = plt.figure()
    plt.plot(guide_points_ct_in_ios[:,0], guide_points_ct_in_ios[:,1], label = 'ct_in_ios', color='red')
    plt.plot(guide_points_ct_to_ios[:,0], guide_points_ct_to_ios[:,1], label = 'ct_to_ios', color='green')
    plt.plot(corrected_guide_points[:, 0], corrected_guide_points[:, 1], label='corrected guided_points', color='blue')
    #plt.plot(test[:,0], test[:,1], label='test')
    plt.legend()

    point1 = corrected_guide_points - guide_points_ct_in_ios
    point2 = guide_points_ct_to_ios - guide_points_ct_in_ios
    point3 = displacement
    print('')
    x = 0
    y = 100
    fig2 = plt.figure()
    plt.plot(range(y), point1[x:y+x, 1]-point3[x:y+x, 1], label='corrected vs. desired motion y')
    #plt.plot(range(20), point2[x:20+x, 0], 'o', label='desired motion')
    plt.plot(range(y), point1[x:y+x, 0]-point3[x:y+x, 0], label='corrected vs. desired motion x')
    plt.plot(range(y), point1[x:y + x, 2] - point3[x:y + x, 2], label='corrected vs. desired motion z')
    plt.legend()

    plt.show()



    fig0 = plt.figure()
    plt.scatter(corrected_spline[:,0], corrected_spline[:,1], label='corrected spline', color='r')
    plt.plot(guide_points_ct_to_ios[:, 0], guide_points_ct_to_ios[:, 1], '-', label='ct_to_ios', color='green')
    #plt.plot(corrected_guide_points[:, 0], corrected_guide_points[:, 1], '+', label='corrected guided_points', color='blue')
    plt.scatter(original_spline[:,0], original_spline[:,1], label='original spline', color='orange')
    plt.plot(corrected_guide_points[:, 0], corrected_guide_points[:, 1], label='corrected guided_points', color='blue')
    plt.legend()


    fig4 = plt.figure()
    plt.scatter(original_spline[:, 0], original_spline[:, 1], label='original spline', color='orange')
    plt.scatter(corrected_spline[:, 0], corrected_spline[:, 1], label='corrected spline', color='red')
    plt.scatter(correction_spline_moved[:, 0], correction_spline_moved[:, 1], label='correction spline moved', color='blue')
    plt.plot(guide_points_ct_to_ios[:, 0], guide_points_ct_to_ios[:, 1], label='ct_to_ios', color='green')
    plt.legend()


    fig3 = plt.figure()
    plt.plot(corrected_guide_points[:,0], corrected_guide_points[:,1], label='corrected guided_points', color='blue')
    plt.plot(guide_points_ct_to_ios[:,0], guide_points_ct_to_ios[:,1], label = 'ct_to_ios', color='green')
    plt.plot(guide_points_ct_in_ios[:, 0], guide_points_ct_in_ios[:, 1], label='ct_in_ios', color='red')
    plt.legend()
    plt.show()

    fig1 = plt.figure()
    plt.scatter(range(len(correction_error)), correction_error)
    plt.scatter(range(len(driving_error)), driving_error)
    #plt.scatter(range(len(verify_error)), verify_error)

    plt.show()

    # Update all points for drawing
    arch_ios.update_all_teeth_points()
    arch_ct_in_ios.update_all_teeth_points()
    arch_ct_to_ios.update_all_teeth_points()
    arch_ct_in_ios_curvilinear_correction.update_all_teeth_points()
    arch_ios_curvilinear_correction.update_all_teeth_points()

    print('checking rigid transformation')
    draw_registration_result_points_array(arch_ios.allpoints, arch_ct_to_ios.allpoints, np.eye(4))
    print('checking local ICP')
    draw_registration_result_points_array(arch_ios.allpoints, arch_ct_in_ios.allpoints, np.eye(4))
    print('checking curvilinear correction')
    draw_registration_result_points_array(arch_ios_curvilinear_correction.allpoints, arch_ct_to_ios.allpoints, np.eye(4))
    print('checking curvilinear correction guided points')
    draw_registration_result_points_array(arch_ct_in_ios_curvilinear_correction.allpoints, arch_ct_to_ios.allpoints, np.eye(4))


