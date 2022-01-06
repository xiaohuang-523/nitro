# Point-surface matching algorithm
# Modified ICP using SVD point registration
# for Cadaver Lab
# by Xiao Huang @ 1/3/2022

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import jstyleson
import csv
import os
import os.path



# Check if file exists
def file_exists(file_path):
    return os.path.isfile(file_path)


# point set registration
# pc1 and pc2 are list of points (could be 2d or 3d points)
# pc2 = R * pc1 + t
def point_set_registration(pc1, pc2):
    pc1_mean = np.mean(np.asarray(pc1), 0)
    pc2_mean = np.mean(np.asarray(pc2), 0)
    pc1_c = np.asarray(pc1) - pc1_mean
    pc2_c = np.asarray(pc2) - pc2_mean
    U,s,V = np.linalg.svd(np.matmul(np.transpose(pc1_c), pc2_c))
    V = np.transpose(V)
    det = np.linalg.det(np.matmul(V,U))
    R = np.matmul(V, np.matmul(np.diag([1, 1, det]), np.transpose(U)))
    t = pc2_mean - np.matmul(R, pc1_mean)

    # check registration FRE
    delta = []
    for p1, p2 in zip(pc1, pc2):
        #print('p1 is', p1)
        #print('p2 is', p2)
        #print('difference is', p2 - np.matmul(R, p1) - t)
        delta.append(np.linalg.norm(p2 - (np.matmul(R, p1) + t))**2)
        #print('delta is', delta)
    FRE = np.sqrt(np.sum(delta)/len(delta))
    #print('fiducial registration FRE is ' + np.str(FRE) + ' mm')
    return R, t


# write csv file with data (matrix)
def write_csv_matrix(file_name, data, fmt = '%.4f', delim = ","):
    f = open(file_name, 'w')
    data = np.asarray(data)
    if len(np.shape(data)) > 1:
        for data_tem in np.asarray(data):
            np.savetxt(f, data_tem.reshape(1, data_tem.shape[0]), delimiter=delim, fmt = fmt)
    else:
        np.savetxt(f, data.reshape(1, len(data)), delimiter=delim, fmt = fmt)
        #f.write("\n")
    f.close()


def plot_3d_points(points, ax, color='red', alpha=1, axe_option = True, unit = 'mm'):
    points = np.asarray(points)
    #ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], zdir='z', s=20, alpha=alpha, c=color,
               rasterized=True)
    # Set up the axes limits
    if axe_option:
        ax.axes.set_xlim3d(left=0, right=300)
        ax.axes.set_ylim3d(bottom=0, top=300)
        ax.axes.set_zlim3d(bottom=0, top=300)

    # Create axes labels
    ax.set_xlabel('X ' + unit)
    ax.set_ylabel('Y ' + unit)
    ax.set_zlabel('Z ' + unit)


def read_csv_specific_rows(fname, jnumber = 7, line_number = [0, 5000], delimiter = ","):
    data = []
    count = 0
    with open(fname, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if count < line_number[0]:
                # pass this row
                count += 1
            else:
                if line_number[1] == -1 or count < line_number[1]:
                    data_tem = []
                    count += 1
                    for idx in range(jnumber):
                        data_tem.append(float(row[idx]))
                    data.append(data_tem)
                else:
                    break
    return np.asarray(data)


# Parse csv files. Information: https://docs.python.org/3/library/csv.html
# flag = 0 will ignore the 1st row.
def read_csv(fname, jnumber = 7, line_number = 50000, flag=1, delimiter = ","):
    bb_tem = np.zeros(jnumber)
    bb_data_t = np.zeros(jnumber)
    #flag = 0
    count = 0
    with open(fname, newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            count += 1
            if flag == 0:
                flag = 1
            elif flag == 1:
                for j in range(jnumber):
                    bb_data_t[j] = np.fromstring(row[j], dtype=float, sep=delimiter)
                bb_tem = np.vstack((bb_tem, bb_data_t))
            if line_number != -1 and count == line_number:
                break
    bb_data_f = bb_tem[1:, :]
    return bb_data_f


def read_YomiSettings(fname, str):
    json_file=open(fname)
    data=jstyleson.load(json_file)
    TACal=np.asarray(data[str])
    return TACal


# Function used to merge all elements in a list
# The elements of a list are np matrices.
# Combine all matrices to make one matrix
def combine_elements_in_list(list):
    matrix = list[0]
    for i in range(1, len(list), 1):
        if list[i] != []:
            matrix = np.vstack((matrix, list[i]))
    return matrix


# Function used to merge all elements in a list
# The elements of a list are np matrices.
# Combine all matrices to make one matrix
def combine_3d_elements_in_list(list):
    matrix = np.zeros(3)
    for i in range(len(list)):
        if list[i] != []:
            matrix = np.vstack((matrix, list[i]))
    return matrix[1:,:]


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
    source_points_array = combine_elements_in_list(source_points_list)
    destination_points_array = combine_elements_in_list(destination_points_list)

    r_, t_ = point_set_registration(source_points_array, destination_points_array)

    source_points_transformed_list = []
    for list_element in source_points_list:
        array_tem = []
        for point in list_element:
            array_tem.append(np.matmul(r_, point) + t_)
        source_points_transformed_list.append(array_tem)
    source_points_transformed_array = combine_elements_in_list(source_points_transformed_list)
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
        self.surface_all_points = combine_3d_elements_in_list(self.points)


if __name__ == '__main__':
    # Define global variables
    SURFACE_LIST = ['lingual', 'occlusal', 'buccal', 'front', 'back']
    CT_FILE_BASE = "C:\\tools probing cadaver lab\\CT raw\\"
    FILE_BASE = "C:\\tools probing cadaver lab\\"
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
    #tooth_list = [17, 18, 21, 24, 25, 28, 31, 32]
    for j in range(1, 33, 1):
        print('j is', j)
        for surface in SURFACE_LIST:
            if file_exists(CT_FILE_BASE + INI + "dicom_points_tooth" + np.str(j) + "_" + surface + ".csv"):
                file_tem = CT_FILE_BASE + INI + "dicom_points_tooth" + np.str(j) + "_" + surface + ".csv"
                print('reading file', file_tem)
                teeth_CT[j-1].add_surface(surface, read_csv(file_tem, 3, -1))
                print('points are', np.shape(read_csv(file_tem, 3, -1)))
                teeth_CT[j - 1].update_all_points()
            if file_exists(CT_FILE_BASE + REG + "dicom_points_tooth" + np.str(j) + "_" + surface + ".csv"):
                file_tem_reg = CT_FILE_BASE + REG + "dicom_points_tooth" + np.str(j) + "_" + surface + ".csv"
                print('reading file', file_tem_reg)
                teeth_CT_reg[j-1].add_surface(surface, read_csv(file_tem_reg, 3, -1))
                print('points are', np.shape(read_csv(file_tem_reg, 3, -1)))
                teeth_CT_reg[j - 1].update_all_points()
        #teeth_CT[j-1].update_all_points()
        #teeth_CT_reg[j-1].update_all_points()

    # read probing points
    probe_points = read_YomiSettings(PROBE_FILE, str='probe_positions') * 1000  # convert m to mm
    probe_surface_name_idx = read_YomiSettings(PROBE_FILE, 'surface_name_idx')
    probe_tooth_number = read_YomiSettings(PROBE_FILE, 'tooth_number')
    print('shape of probe points are', np.shape(probe_points))
    print('surface_name_idx is', probe_surface_name_idx)
    print('tooth_number is', probe_tooth_number)
    for i in range(len(probe_tooth_number)):
        tooth_idx_tem = probe_tooth_number[i] - 1
        surface_idx_tem = probe_surface_name_idx[i] - 1
        surface_name_tem = SURFACE_LIST[surface_idx_tem]
        points_tem = probe_points[i]
        teeth_Probe[tooth_idx_tem].add_surface(surface_name_tem, points_tem)
        print('reading tooth ', tooth_idx_tem + 1, ' surface ', surface_name_tem)
        teeth_Probe[tooth_idx_tem].update_all_points()

    centers_CT = []
    centers_Probe = []
    surfaces_CT = []
    surfaces_Probe = []

    register_teeth_idx_list = []
    register_surface_idx_list = []

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
        # REG CT surfaces contain more points to assure that the probing points on certain surface are always covered.
        surfaces_CT.append(teeth_CT_reg[n_tem].points[surface_idx_tem])
        centers_Probe.append(teeth_Probe[n_tem].surface_centers[surface_idx_tem])
        surfaces_Probe.append(teeth_Probe[n_tem].points[surface_idx_tem])

    centers_CT = combine_elements_in_list(centers_CT)
    centers_Probe = combine_elements_in_list(centers_Probe)
    original_points_Probe = combine_elements_in_list(surfaces_Probe)

    print(' ')
    print('All surfaces are found, performing registration')
    # Perform initial alignment
    R, t = point_set_registration(centers_Probe, centers_CT)
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

    R_final, t_final = point_set_registration(original_points_Probe, combine_elements_in_list(source_points))
    print('Registraion is finished, plotting results')
    #print('source points are', combine_elements_in_list(source_points))

    # Plotting results
    surfaces_CT = combine_elements_in_list(surfaces_CT)
    #print('surfaces CT are', surfaces_CT)
    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')
    plot_3d_points(surfaces_CT, ax, color='green', alpha=0.2, axe_option=False)
    plot_3d_points(combine_elements_in_list(source_points), ax, color='red', axe_option=False)
    plt.show()

    FIDUCIAL_ARRAY_FS_FILE = "C:\\Neocis\\FiducialArrays\\FXT-0086-07-LRUL-MFG-Splint.txt"
    fiducial_array_fs = read_csv_specific_rows(FIDUCIAL_ARRAY_FS_FILE, 4, [3, -1], delimiter=' ')[:,1:]
    fiducial_array_ct = []
    fiducial_array_ct_2 = []
    for point in fiducial_array_fs:
        fiducial_array_ct_2.append(np.matmul(R_final, point) + t_final)
        fiducial_array_ct.append(np.matmul(np.linalg.inv(R_final), (point-t_final)))
    fiducial_array_ct = np.asarray(fiducial_array_ct)
    fiducial_array_ct_2 = np.asarray(fiducial_array_ct_2)
    FIDUCIAL_ARRAY_CT_FILE = FILE_BASE + "\\fiducials_array_cadaver_lab.txt"
    write_csv_matrix(FIDUCIAL_ARRAY_CT_FILE, fiducial_array_ct_2, fmt='%.6f', delim=' ')

