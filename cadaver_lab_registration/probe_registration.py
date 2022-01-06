# Point-surface matching algorithm
# Modified ICP using SVD point registration
# for Cadaver Lab
# by Xiao Huang @ 1/3/2022

import numpy as np
import Readers as Yomiread
import plot as Yomiplot
import matplotlib.pyplot as plt
import array_processing as ap
import registration
import open3d as o3d
import fileDialog
import Writers as Yomiwrite


def point_surface_distance(point, surface):
    dis_abs = np.linalg.norm(point - surface[0,:])
    # dis_vec = point - surface[0,:]
    selected_point = surface[0,:]
    for point_tem in surface:
        dis_tem = np.linalg.norm(point - point_tem)
        if dis_tem < dis_abs:
            dis_abs = dis_tem
            # dis_vec = point - point_tem
            del selected_point
            selected_point = point_tem
    return dis_abs, selected_point


def select_closest_point(point_list, surface):
    selected_points = []
    for point in point_list:
        dis_abs, point_tem = point_surface_distance(point, surface)
        selected_points.append(point_tem)
    return np.asarray(selected_points)


def select_closest_point_on_tooth(probing_points_transformed, buccal, front, lingual, occlusal):
    buccal_selected_points = select_closest_point(probing_points_transformed[0:3,:], buccal)
    front_selected_points = select_closest_point(probing_points_transformed[3:6,:], front)
    lingual_selected_points = select_closest_point(probing_points_transformed[6:9,:], lingual)
    occlusal_selected_points = select_closest_point(probing_points_transformed[9:12,:], occlusal)
    return buccal_selected_points, front_selected_points


def modified_ICP(Probe_surfaces_list, CT_surfaces_list):
    source_points_list = Probe_surfaces_list
    destination_points_list = []
    for i in range(len(Probe_surfaces_list)):
        closet_points = select_closest_point(Probe_surfaces_list[i], CT_surfaces_list[i])
        destination_points_list.append(closet_points)
    source_points_array = ap.combine_elements_in_list(source_points_list)
    destination_points_array = ap.combine_elements_in_list(destination_points_list)

    r_, t_ = registration.point_set_registration(source_points_array, destination_points_array)

    source_points_transformed_list = []
    for list_element in source_points_list:
        array_tem = []
        for point in list_element:
            array_tem.append(np.matmul(r_, point) + t_)
        source_points_transformed_list.append(array_tem)
    source_points_transformed_array = ap.combine_elements_in_list(source_points_transformed_list)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #Yomiplot.plot_3d_points(ios_points_transformed_2,ax,color='red')
    #Yomiplot.plot_3d_points(destination_points, ax, color='green')
    #Yomiplot.plot_3d_points(source_points, ax, color='blue')
    #Yomiplot.plot_3d_points(occlusal,ax,color='red', alpha=0.2)
    #plt.show()

    error = np.linalg.norm(source_points_transformed_array - destination_points_array, axis=1)
    error = np.sqrt(np.sum(error**2)/len(error))
    print('error rmse is', error)
    return error, source_points_transformed_list, destination_points_list


# Tooth features, explain from which space and in which space
class tooth:
    def __init__(self, tooth_id, in_space='CT'):
        self.lingual_points = []
        self.buccal_points = []
        self.occlusal_points = []
        self.front_points = []
        self.back_points = []
        #self.points = [self.lingual_points, self.occlusal_points, self.buccal_points, self.front_points, self.back_points]
        self.points = []
        self.surface_all_points = []

        self.lingual_center = []
        self.buccal_center = []
        self.occlusal_center = []
        self.front_center = []
        self.back_center = []
        self.surface_centers = []

        self.id = int(tooth_id)
        self.in_space = in_space
        self.initial_transformation = []
        self.modified_ICP = []
        self.local_ICP_transformation = []

    def add_surface(self, surface_name_str, n_array_n_by_3):
        if surface_name_str == 'lingual':
            self.lingual_points = n_array_n_by_3
            self.lingual_center = np.sum(n_array_n_by_3, axis=0)/len(n_array_n_by_3)
        elif surface_name_str == 'occlusal':
            self.occlusal_points = n_array_n_by_3
            self.occlusal_center = np.sum(n_array_n_by_3, axis=0) / len(n_array_n_by_3)
        elif surface_name_str == 'buccal':
            self.buccal_points = n_array_n_by_3
            self.buccal_center = np.sum(n_array_n_by_3, axis=0) / len(n_array_n_by_3)
        elif surface_name_str == 'front':
            self.front_points = n_array_n_by_3
            self.front_center = np.sum(n_array_n_by_3, axis=0) / len(n_array_n_by_3)
        elif surface_name_str == 'back':
            self.back_points = n_array_n_by_3
            self.back_center = np.sum(n_array_n_by_3, axis=0) / len(n_array_n_by_3)
        else:
            raise TypeError('wrong surface type, check surface_name_str')

    def update_all_points(self):
        self.points = [self.lingual_points, self.occlusal_points, self.buccal_points, self.front_points,
                       self.back_points]
        self.surface_centers = [self.lingual_center, self.occlusal_center, self.buccal_center, self.front_center,
                                self.back_center]
        self.surface_all_points = ap.combine_3d_elements_in_list(self.points)


if __name__ == '__main__':
    # Define global variables
    SURFACE_LIST = ['lingual', 'occlusal', 'buccal', 'front', 'back']
    CT_FILE_BASE = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\HW-9898 Point-to-surface experiments\\CT raw\\"
    REG = "registration_selection_pc\\"
    INI = "initial_alignment_pc\\"
    PROBE_FILE = "C:\\Neocis\\Output\\GAPTRegistrationData\\gapt-registration-data\\final_probe_poses.json"

    # Initialize two classes
    teeth_CT = []
    teeth_CT_reg = []
    teeth_Probe = []

    for j in range(32):
        teeth_CT.append(tooth(j+1, 'CT'))
        teeth_CT_reg.append(tooth(j+1, 'CT'))
        teeth_Probe.append(tooth(j+1, 'Probe'))

    # read CT surface points
    tooth_list = [17, 18, 21, 24, 25, 28, 31, 32]
    for j in tooth_list:
        for surface in SURFACE_LIST:
            if fileDialog.file_exists(CT_FILE_BASE + INI + "dicom_points_tooth" + np.str(j) + "_" + surface + ".csv"):
                file_tem = CT_FILE_BASE + INI + "dicom_points_tooth" + np.str(j) + "_" + surface + ".csv"
                print('reading file', file_tem)
                teeth_CT[j-1].add_surface(surface, Yomiread.read_csv(file_tem, 3, -1))
                print('points are', np.shape(Yomiread.read_csv(file_tem, 3, -1)))
            if fileDialog.file_exists(CT_FILE_BASE + REG + "dicom_points_tooth" + np.str(j) + "_" + surface + ".csv"):
                file_tem_reg = CT_FILE_BASE + REG + "dicom_points_tooth" + np.str(j) + "_" + surface + ".csv"
                print('reading file', file_tem_reg)
                teeth_CT_reg[j-1].add_surface(surface, Yomiread.read_csv(file_tem_reg, 3, -1))
                print('points are', np.shape(Yomiread.read_csv(file_tem_reg, 3, -1)))
        teeth_CT[j-1].update_all_points()
        teeth_CT_reg[j-1].update_all_points()

    # read probing points
    probe_points = Yomiread.read_YomiSettings(PROBE_FILE, str='probe_positions') * 1000  # convert m to mm
    probe_surface_name_idx = Yomiread.read_YomiSettings(PROBE_FILE, 'surface_name_idx')
    probe_tooth_number = Yomiread.read_YomiSettings(PROBE_FILE, 'tooth_number')
    print('shape of probe points are', np.shape(probe_points))
    print('surface_name_idx is', probe_surface_name_idx)
    print('tooth_number is', probe_tooth_number)
    for i in range(len(probe_tooth_number)):
        tooth_idx_tem = probe_tooth_number[i] - 1
        surface_idx_tem = probe_surface_name_idx[i] - 1
        surface_name_tem = SURFACE_LIST[surface_idx_tem]
        #print('surface_idx_tem is', surface_idx_tem)
        #print('surface_name_tem is', surface_name_tem)
        points_tem = probe_points[i]
        teeth_Probe[tooth_idx_tem].add_surface(surface_name_tem, points_tem)
        print('reading tooth ', tooth_idx_tem + 1, ' surface ', surface_name_tem)
        teeth_Probe[tooth_idx_tem].update_all_points()

    #probe_teeth_list = [24, 21]
    #register_teeth_list = [24, 21, 24, 24, 21, 21]
    # select surface from list 'lingual', 'occlusal', 'buccal', 'front', 'back'
    #register_surface_list = ['occlusal', 'occlusal', 'lingual', 'buccal', 'buccal', 'lingual']

    centers_CT = []
    centers_Probe = []
    surfaces_CT = []
    surfaces_Probe = []

    register_teeth_idx_list = []
    register_surface_idx_list = []
    #all_teeth_surfaces = 32*5
    #select_register_teeth_flag = True

    registration_surface_amount = np.int(input('How many surfaces in total are used for registration? \n'))
    print('Specify the tooth number and surface name. \n')

    for i in range(registration_surface_amount):
        print('select registration tooth surface ', i+1)
        n_tem = np.int(input("Enter the register tooth number: \n")) - 1
        register_teeth_idx_list.append(n_tem)

        select_register_surface_flag = True
        while select_register_surface_flag:
            surface_tem = input("Enter the register tooth surface \n'l' for lingual, 'o' for occlusal, 'b' for buccal, 'f' for front and 'k' for back. \n")
            if surface_tem == 'l':
                surface_idx_tem = 0
                select_register_surface_flag = False
            elif surface_tem == 'o':
                surface_idx_tem = 1
                select_register_surface_flag = False
            elif surface_tem == 'b':
                surface_idx_tem = 2
                select_register_surface_flag = False
            elif surface_tem == 'f':
                surface_idx_tem = 3
                select_register_surface_flag = False
            elif surface_tem == 'k':
                surface_idx_tem = 4
                select_register_surface_flag = False
            else:
                print('Wrong surface name was provided')
        register_surface_idx_list.append(surface_idx_tem)
        print('Registration surface '+ np.str(i+1) + ' is tooth '+ np.str(n_tem + 1) + ' surface '+SURFACE_LIST[surface_idx_tem])
        # INI CT surfaces are well tuned surfaces which give more accurate center points.
        centers_CT.append(teeth_CT[n_tem].surface_centers[surface_idx_tem])
        #print('center_CT is', centers_CT)
        # REG CT surfaces contain more points to assure that the probing points on certain surface are always covered.
        surfaces_CT.append(teeth_CT_reg[n_tem].points[surface_idx_tem])
        centers_Probe.append(teeth_Probe[n_tem].surface_centers[surface_idx_tem])
        surfaces_Probe.append(teeth_Probe[n_tem].points[surface_idx_tem])

    # for i in range(len(probe_tooth_number)):
    #     tooth_idx_tem = probe_tooth_number[i] - 1
    #     surface_idx_tem = probe_surface_name_idx[i] - 1
    #     surface_name_tem = SURFACE_LIST[surface_idx_tem]
    #     #print('surface_idx_tem is', surface_idx_tem)
    #     #print('surface_name_tem is', surface_name_tem)
    #     points_tem = probe_points[i]
    #     teeth_Probe[tooth_idx_tem].add_surface(surface_name_tem, points_tem)
    #     print('reading tooth ', tooth_idx_tem + 1, ' surface ', surface_name_tem)
    #     teeth_Probe[tooth_idx_tem].update_all_points()
    #
    #     # INI CT surfaces are well tuned surfaces which give more accurate center points.
    #     centers_CT.append(teeth_CT[tooth_idx_tem].surface_centers[surface_idx_tem])
    #     print('center_CT is', centers_CT)
    #     # REG CT surfaces contain more points to assure that the probing points on certain surface are always covered.
    #     surfaces_CT.append(teeth_CT_reg[tooth_idx_tem].points[surface_idx_tem])
    #     centers_Probe.append(teeth_Probe[tooth_idx_tem].surface_centers[surface_idx_tem])
    #     surfaces_Probe.append(teeth_Probe[tooth_idx_tem].points[surface_idx_tem])

    centers_CT = ap.combine_elements_in_list(centers_CT)
    #surfaces_CT = ap.combine_elements_in_list(surfaces_CT)
    centers_Probe = ap.combine_elements_in_list(centers_Probe)
    original_points_Probe = ap.combine_elements_in_list(surfaces_Probe)

    print('shape of centers CT is', np.shape(centers_CT))
    print('shape of centers Probe is', np.shape(centers_Probe))
    print('shape of surfaces CT is', np.shape(surfaces_CT))
    print('shape of surfaces Probe is', np.shape(surfaces_Probe))

    print(' ')
    print('All surfaces are found, performing registration')
    # Perform initial alignment
    R, t = registration.point_set_registration(centers_Probe, centers_CT)
    surfaces_Probe_transformed = []
    for surface in surfaces_Probe:
        points_tem = []
        for point in surface:
            points_tem.append(np.matmul(R, point) + t)
        points_tem = np.asarray(points_tem)
        surfaces_Probe_transformed.append(points_tem)
        del points_tem

    # Perform registration
    source_points = surfaces_Probe_transformed
    for k in range(50):
        error, source_points_new, destination_points_new = modified_ICP(source_points, surfaces_CT)
        del source_points
        source_points = source_points_new
        # print('destination_points are', destination_points_new[0,:])
        if error < 0.01:
            break

    print('shape of original robot points is', np.shape(original_points_Probe))
    print('shape of final robot points is', np.shape(ap.combine_elements_in_list(source_points)))
    R_final, t_final = registration.point_set_registration(original_points_Probe, ap.combine_elements_in_list(source_points))

    # Plotting results
    plot_tooth = 21
    source_points_array = ap.combine_elements_in_list(source_points)
    surfaces_CT = ap.combine_elements_in_list(surfaces_CT)
    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')
    Yomiplot.plot_3d_points(teeth_CT[plot_tooth-1].occlusal_points, ax, color='green', alpha=0.2, axe_option=False)
    Yomiplot.plot_3d_points(source_points_array[4:8, :], ax, color='red', axe_option=False)
    #Yomiplot.plot_3d_points(source_points, ax, color='blue')
    plt.show()


    FIDUCIAL_ARRAY_FS_FILE = "C:\\Neocis\\FiducialArrays\\FXT-0086-07-LRUL-MFG-Splint.txt"
    fiducial_array_fs = Yomiread.read_csv_specific_rows(FIDUCIAL_ARRAY_FS_FILE, 4, [3, -1], delimiter=' ')[:,1:]
    fiducial_array_ct = []
    fiducial_array_ct_2 = []
    for point in fiducial_array_fs:
        fiducial_array_ct_2.append(np.matmul(R_final, point) + t_final)
        fiducial_array_ct.append(np.matmul(np.linalg.inv(R_final), (point-t_final)))
    fiducial_array_ct = np.asarray(fiducial_array_ct)
    fiducial_array_ct_2 = np.asarray(fiducial_array_ct_2)
    FIDUCIAL_ARRAY_CT_FILE = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous" \
                             "\\HW-9898 Point-to-surface experiments\\Yomiplan Case File\\fiducials_array_ct.txt"
    FIDUCIAL_ARRAY_CT_FILE_2 = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous" \
                             "\\HW-9898 Point-to-surface experiments\\Yomiplan Case File\\fiducials_array_ct_2.txt"

    Yomiwrite.write_csv_matrix(FIDUCIAL_ARRAY_CT_FILE, fiducial_array_ct, fmt='%.6f', delim=' ')
    Yomiwrite.write_csv_matrix(FIDUCIAL_ARRAY_CT_FILE_2, fiducial_array_ct_2, fmt='%.6f', delim=' ')

    exit()







    exit()
    # tooth is probed in the order of lingual, occlusal, buccal, front, back
    # hardcoded for probing teeth for testing
    probe_teeth_surface = [['lingual', 'occlusal', 'buccal', 'back'],
                           ['lingual', 'occlusal', 'buccal', 'front'],
                           ['lingual', 'occlusal', 'buccal', 'front', 'back'],
                           ['lingual', 'buccal', 'occlusal'],
                           ['lingual', 'buccal', 'occlusal'],
                           ['lingual', 'occlusal', 'buccal', 'front', 'back'],
                           ['lingual', 'occlusal', 'buccal', 'front'],
                           ['lingual', 'occlusal', 'buccal', 'back']]

    probe_teeth_list = [24, 21]
    register_teeth_list = [24, 21, 24, 24, 21, 21]
    # select surface from list 'lingual', 'occlusal', 'buccal', 'front', 'back'
    register_surface_list = ['occlusal', 'occlusal', 'lingual', 'buccal', 'buccal', 'lingual']


    #probe_tooth_list = [18, 28, 31]
    number_of_probe_surface = 0
    #number_of_points_per_surface = 4
    for i in range(len(register_teeth_list)):
        if i < 2:
            number_of_points_per_surface = 6
            row_start = i * number_of_points_per_surface
            row_end = (i + 1) * number_of_points_per_surface
        else:
            number_of_points_per_surface = 3
            row_start = i * number_of_points_per_surface + 2*(6-number_of_points_per_surface)
            row_end = (i + 1) * number_of_points_per_surface + 2*(6-number_of_points_per_surface)
        idx = register_teeth_list[i]
        surface_name = register_surface_list[i]
        print('surface name is', surface_name)
        print('tooth idx is', idx)
        #row_start = i * number_of_points_per_surface
        #row_end = (i + 1) * number_of_points_per_surface
        teeth_Probe[idx-1].add_surface(surface_name, probe_points[row_start:row_end,:])
        print('reading tooth ', idx, ' surface ', surface_name , 'from rows ', row_start)
        #print('shape of points is', np.shape(teeth_Probe[idx-1].buccal_points))
    for tooth in probe_teeth_list:
        teeth_Probe[tooth-1].update_all_points()


    # Check all surface points

    #all_points = []
    #for list in tooth_list:
    #    print('shape is', np.shape(teeth_CT[list-1].surface_all_points))
    #    all_points.append(teeth_CT[list-1].surface_all_points)
    #all_points = ap.combine_elements_in_list(all_points)

    #for i in tooth_list:
        #print('surface centers are ', teeth_CT[i-1].surface_centers)
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #Yomiplot.plot_3d_points(teeth_CT[i-1].surface_all_points, ax, color='green')
        #plt.show()

    # Registration
    # Prepare data for registration
    #register_teeth_list = [18, 18, 24, 28, 32, 32]
    # select surface from list 'lingual', 'occlusal', 'buccal', 'front', 'back'
    #register_surface_list = ['buccal', 'occlusal', 'buccal', 'occlusal', 'buccal', 'back']

    centers_CT = []
    centers_Probe = []
    surfaces_CT = []
    surfaces_Probe = []
    #for
    for i in range(len(register_teeth_list)):
        idx = register_teeth_list[i]
        # check surface idx based on ['lingual', 'occlusal', 'buccal', 'front', 'back']
        surface = register_surface_list[i]
        if surface == 'lingual':
            surface_idx = 0
        elif surface == 'occlusal':
            surface_idx = 1
        elif surface == 'buccal':
            surface_idx = 2
        elif surface == 'front':
            surface_idx = 3
        elif surface == 'back':
            surface_idx = 4
        print('surface is', surface)
        print('surface idx is', surface_idx)
        print('tooth idx is', idx)
        # INI CT surfaces are well tuned surfaces which give more accurate center points.
        centers_CT.append(teeth_CT[idx-1].surface_centers[surface_idx])
        print('center_CT is', centers_CT)
        # REG CT surfaces contain more points to assure that the probing points on certain surface are always covered.
        surfaces_CT.append(teeth_CT_reg[idx-1].points[surface_idx])
        centers_Probe.append(teeth_Probe[idx-1].surface_centers[surface_idx])
        surfaces_Probe.append(teeth_Probe[idx-1].points[surface_idx])


    centers_CT = ap.combine_elements_in_list(centers_CT)
    #surfaces_CT = ap.combine_elements_in_list(surfaces_CT)
    centers_Probe = ap.combine_elements_in_list(centers_Probe)
    original_points_Probe = ap.combine_elements_in_list(surfaces_Probe)

    print('shape of centers CT is', np.shape(centers_CT))
    print('shape of centers Probe is', np.shape(centers_Probe))
    print('shape of surfaces CT is', np.shape(surfaces_CT))
    print('shape of surfaces Probe is', np.shape(surfaces_Probe))

    # Perform initial alignment
    R, t = registration.point_set_registration(centers_Probe, centers_CT)
    surfaces_Probe_transformed = []
    for surface in surfaces_Probe:
        points_tem = []
        for point in surface:
            points_tem.append(np.matmul(R, point) + t)
        points_tem = np.asarray(points_tem)
        surfaces_Probe_transformed.append(points_tem)
        del points_tem

    # Perform registration
    source_points = surfaces_Probe_transformed
    for k in range(50):
        error, source_points_new, destination_points_new = modified_ICP(source_points, surfaces_CT)
        del source_points
        source_points = source_points_new
        # print('destination_points are', destination_points_new[0,:])
        if error < 0.01:
            break

    print('shape of original robot points is', np.shape(original_points_Probe))
    print('shape of final robot points is', np.shape(ap.combine_elements_in_list(source_points)))
    R_final, t_final = registration.point_set_registration(original_points_Probe, ap.combine_elements_in_list(source_points))

    # Plotting results
    plot_tooth = 21
    source_points_array = ap.combine_elements_in_list(source_points)
    surfaces_CT = ap.combine_elements_in_list(surfaces_CT)
    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')
    Yomiplot.plot_3d_points(teeth_CT[plot_tooth-1].occlusal_points, ax, color='green', alpha=0.2)
    Yomiplot.plot_3d_points(source_points_array[4:8, :], ax, color='red')
    #Yomiplot.plot_3d_points(source_points, ax, color='blue')
    plt.show()


    FIDUCIAL_ARRAY_FS_FILE = "C:\\Neocis\\FiducialArrays\\FXT-0086-07-LRUL-MFG-Splint.txt"
    fiducial_array_fs = Yomiread.read_csv_specific_rows(FIDUCIAL_ARRAY_FS_FILE, 4, [3, -1], delimiter=' ')[:,1:]
    fiducial_array_ct = []
    fiducial_array_ct_2 = []
    for point in fiducial_array_fs:
        fiducial_array_ct_2.append(np.matmul(R_final, point) + t_final)
        fiducial_array_ct.append(np.matmul(np.linalg.inv(R_final), (point-t_final)))
    fiducial_array_ct = np.asarray(fiducial_array_ct)
    fiducial_array_ct_2 = np.asarray(fiducial_array_ct_2)
    FIDUCIAL_ARRAY_CT_FILE = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous" \
                             "\\HW-9898 Point-to-surface experiments\\Yomiplan Case File\\fiducials_array_ct.txt"
    FIDUCIAL_ARRAY_CT_FILE_2 = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous" \
                             "\\HW-9898 Point-to-surface experiments\\Yomiplan Case File\\fiducials_array_ct_2.txt"

    Yomiwrite.write_csv_matrix(FIDUCIAL_ARRAY_CT_FILE, fiducial_array_ct, fmt='%.6f', delim=' ')
    Yomiwrite.write_csv_matrix(FIDUCIAL_ARRAY_CT_FILE_2, fiducial_array_ct_2, fmt='%.6f', delim=' ')

    exit()
