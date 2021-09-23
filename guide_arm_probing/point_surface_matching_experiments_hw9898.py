# Developing point-surface matching algorithm
# Modified ICP using SVD point registration
# Key point is to select points from know surfaces
# by Xiao Huang @ 09/10/2021

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
        self.surface_all_points = ap.combine_elements_in_list(self.points)


if __name__ == '__main__':
    # Define global variables
    SURFACE_LIST = ['lingual', 'occlusal', 'buccal', 'front', 'back']
    CT_FILE_BASE = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\HW-9898 Point-to-surface experiments\\CT raw\\"
    PROBE_FILE = "C:\\Neocis\\Output\\GAPTRegistrationData\\gapt-registration-data\\final_probe_poses.json"

    # Initialize two classes
    teeth_CT = []
    teeth_Probe = []

    for j in range(32):
        teeth_CT.append(tooth(j+1, 'CT'))
        teeth_Probe.append(tooth(j+1, 'Probe'))

    # read CT surface points
    tooth_list = [17, 18, 21, 24, 25, 28, 31, 32]
    for j in tooth_list:
        for surface in SURFACE_LIST:
            if fileDialog.file_exists(CT_FILE_BASE + "dicom_points_tooth" + np.str(j) + "_" + surface + ".csv"):
                file_tem = CT_FILE_BASE + "dicom_points_tooth" + np.str(j) + "_" + surface + ".csv"
                print('reading file', file_tem)
                teeth_CT[j-1].add_surface(surface, Yomiread.read_csv(file_tem, 3, -1))
                print('points are', np.shape(Yomiread.read_csv(file_tem, 3, -1)))
        teeth_CT[j-1].update_all_points()

    # read probing points
    probe_points = Yomiread.read_YomiSettings(PROBE_FILE, str='probe_positions') * 1000  # convert m to mm
    print('shape of probe points are', np.shape(probe_points))


    # tooth is probed in the order of lingual, occlusal, buccal, front, back
    # hardcoded for probing teeth for testing
    probe_teeth_surface = [['lingual', 'occlusal', 'buccal', 'back'],
                           ['lingual', 'occlusal', 'buccal', 'front'],
                           ['lingual', 'occlusal', 'buccal', 'front', 'back'],
                           ['lingual', 'buccal'],
                           ['lingual', 'buccal'],
                           ['lingual', 'occlusal', 'buccal', 'front', 'back'],
                           ['lingual', 'occlusal', 'buccal', 'front'],
                           ['lingual', 'occlusal', 'buccal', 'back']]
    probe_tooth_list = [18, 28, 31]
    number_of_probe_surface = 0
    for i in probe_tooth_list:
        idx = tooth_list.index(i)
        surface_tem = probe_teeth_surface[idx]
        for probed_surface in surface_tem:
            row_start = number_of_probe_surface * 3
            row_end = (number_of_probe_surface + 1) * 3
            teeth_Probe[i-1].add_surface(probed_surface, probe_points[row_start:row_end,:])
            print('reading tooth ', i, ' surface ', probed_surface , 'from rows ', row_start)
            number_of_probe_surface += 1
        teeth_Probe[i-1].update_all_points()
        del surface_tem

    # Check all surface points

    all_points = []
    for list in tooth_list:
        print('shape is', np.shape(teeth_CT[list-1].surface_all_points))
        all_points.append(teeth_CT[list-1].surface_all_points)
    all_points = ap.combine_elements_in_list(all_points)
    #for i in tooth_list:
        #print('surface centers are ', teeth_CT[i-1].surface_centers)
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #Yomiplot.plot_3d_points(teeth_CT[i-1].surface_all_points, ax, color='green')
        #plt.show()

    # Registration
    # Prepare data for registration
    register_teeth_list = [18, 28, 31]
    plot_tooth = 18
    centers_CT = []
    centers_Probe = []
    surfaces_CT = []
    surfaces_Probe = []
    #for
    for i in register_teeth_list:
        idx = tooth_list.index(i)
        surface_list = probe_teeth_surface[idx]
        for surface in surface_list:
            surface_idx = SURFACE_LIST.index(surface)
            print('surface is', surface)
            print('surface idx is', surface_idx)
            centers_CT.append(teeth_CT[i-1].surface_centers[surface_idx])
            surfaces_CT.append(teeth_CT[i-1].points[surface_idx])
            print('i is', i)
            print('shape of teeth_Probe is', np.shape(teeth_Probe))
            centers_Probe.append(teeth_Probe[i-1].surface_centers[surface_idx])
            surfaces_Probe.append(teeth_Probe[i-1].points[surface_idx])

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

    source_points_array = ap.combine_elements_in_list(source_points)
    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')
    Yomiplot.plot_3d_points(teeth_CT[plot_tooth-1].surface_all_points, ax, color='green', alpha=0.2)
    Yomiplot.plot_3d_points(source_points_array, ax, color='red')
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
