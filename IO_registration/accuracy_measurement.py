import numpy as np
import Readers as Yomiread
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import Axes3D
import Writers as Yomiwrite


# Function to process fixture raw measurements.
# Use the raw cylinder data to define a reference frame and calculate the cylinder base positions
# Input:
#           raw: IOS cylinders measurement raw data
#           original_cylinder_number: select which cylinder is used to specify the origin and z axis
#           Yaxis_cylinder_number: select which cylinder is used to calculate the Y axis
# Output:
#           List of cylinder base positions in the defined reference frame
def check_cylinder_base(raw, origin_cylinder_number, Yaxis_cylinder_number):
    # select cylinders to define the reference frame
    cylinder_origin = origin_cylinder_number     # select the 5th cylinder as origin cylinder
    cylinder_Yaxis = Yaxis_cylinder_number      # select the 3rd cylinder to define Y axis

    # calculate reference frame
    origin = raw[cylinder_origin-1, 0:3]
    z_axis = -raw[cylinder_origin-1, 3:6]   # make sure the cylinder axis is point up
    y_axis_tem = raw[cylinder_Yaxis-1, 0:3] - raw[cylinder_origin-1, 0:3]
    x_axis_tem = np.cross(y_axis_tem, z_axis)
    x_axis = x_axis_tem / np.linalg.norm(x_axis_tem)
    y_axis = np.cross(z_axis,x_axis)

    # calculate transformation matrix
    rot = np.eye(3)
    rot[:,0] = x_axis
    rot[:,1] = y_axis
    rot[:,2] = z_axis
    trans = origin
    T = np.hstack((rot, trans.reshape((3,1))))
    T = np.vstack((T, np.array([0, 0, 0, 1])))

    # calculate cylinder base positions in the reference frame
    cylinder_base = []
    for cylinder in raw:
        pos = np.insert(cylinder[0:3], 3, 1)
        print('pos is', pos)
        cylinder_base_tem = np.matmul(np.linalg.inv(T), pos)[0:3]
        cylinder_base.append(cylinder_base_tem)
    return np.asarray(cylinder_base)


# Function to process fixture raw measurements.
# Use the raw cylinder data to define a reference frame and calculate the cylinder base positions for full arch
# Input:
#           raw: IOS cylinders measurement raw data
#           origin_cylinder_number: select which cylinder is used to specify the origin and z axis. (count from 1)
#           Xaxis_cylinder_number: select which cylinder is used to calculate the X axis (count from 1)
#                                   x axis is pointing from Xaxis_cylinder to origin_cylinder
#           axis: which axis to flip (makeing sure it's positive)
# Output:
#           List of cylinder base positions in the defined reference frame
def check_cylinder_base_full(raw, origin_cylinder_number, Xaxis_cylinder_number, axis='z', reference = 'plane'):
    # select cylinders to define the reference frame
    cylinder_origin = origin_cylinder_number     # select the 5th cylinder as origin cylinder
    cylinder_Xaxis = Xaxis_cylinder_number      # select the 3rd cylinder to define Y axis

    # calculate reference frame
    origin = raw[cylinder_origin-1, 0:3]
    if reference == 'cylinder':
        z_axis = flip_vector(raw[cylinder_origin-1, 3:6], axis)   # make sure the cylinder axis is point up
    if reference == 'plane':
        z_axis = flip_vector(raw[-1, 3:6], axis)  # make sure the cylinder axis is point up

    x_axis_tem = raw[cylinder_origin-1, 0:3] - raw[cylinder_Xaxis-1, 0:3]
    y_axis_tem = np.cross(z_axis, x_axis_tem)
    y_axis = y_axis_tem / np.linalg.norm(y_axis_tem)
    x_axis = np.cross(y_axis,z_axis)
    # calculate transformation matrix
    rot = np.eye(3)
    rot[:,0] = x_axis
    rot[:,1] = y_axis
    rot[:,2] = z_axis
    trans = origin
    T = np.hstack((rot, trans.reshape((3,1))))
    T = np.vstack((T, np.array([0, 0, 0, 1])))
    print('rot is', rot)

    # calculate cylinder base positions in new reference frame
    cylinder_base = []
    for cylinder in raw:
        pos = np.insert(cylinder[0:3], 3, 1)
        print('pos is', pos)
        cylinder_base_tem = np.matmul(np.linalg.inv(T), pos)[0:3]
        cylinder_base.append(cylinder_base_tem)
    # calculate cylinder axis in new reference frame
    cylinder_axis = []
    for cylinder in raw:
        axis_tem = cylinder[3:6]
        cylinder_axis_tem = np.matmul(np.linalg.inv(rot), axis_tem)
        cylinder_axis.append(flip_vector(cylinder_axis_tem))
    return np.asarray(cylinder_base[0:-1]), np.asarray(cylinder_axis[0:-1])


# Function to process fixture raw measurements.
# Use the raw cylinder data to calculate the cylinder orientation errors
# Input:
#           raw: IOS cylinders measurement raw data
# Output:
#           List of cylinder axis errors w.r.t the 1st cylinder
def check_cylinder_orientation(raw):
    ref_axis = raw[0, 3:6]
    angular_error = []
    for cylinder in raw:
        angular_error_tem = np.inner(ref_axis, cylinder[3:6])
        angular_error_tem = angular_error_tem /(np.linalg.norm(ref_axis) * np.linalg.norm(cylinder[3:6]))
        angular_error_tem = np.arccos(angular_error_tem) * 180/np.pi
        angular_error.append(angular_error_tem)
    return np.asarray(angular_error)


# Function to process fixture raw measurements.
# Use the raw cylinder data to calculate the cylinder orientation errors
# Input:
#           raw: IOS cylinders measurement raw data
# Output:
#           List of cylinder axis errors w.r.t the 1st cylinder
def check_cylinder_orientation_full_relative(raw):
    ref_axis = np.array([0,0,1])
    angular_error = []
    for cylinder in raw:
        axis = flip_vector(cylinder)
        angular_error_tem = np.inner(ref_axis, axis)
        angular_error_tem = angular_error_tem /(np.linalg.norm(ref_axis) * np.linalg.norm(axis))
        angular_error_tem = np.arccos(angular_error_tem) * 180/np.pi
        if angular_error_tem > 90:
            angular_error_tem = 180 - angular_error_tem
        angular_error.append(angular_error_tem)
    return np.asarray(angular_error)


# Function to process fixture raw measurements.
# Use the raw cylinder data to calculate the cylinder orientation errors
# Input:
#           raw: IOS cylinders measurement raw data
# Output:
#           List of cylinder axis errors w.r.t the 1st cylinder
def check_cylinder_orientation_full_absolute(raw, faro_converted):
    angular_error = []
    for j in range(7):
        axis = flip_vector(raw[j,:])
        ref = faro_converted[j,:]
        print('raw is', raw)
        print('shape of axis is', axis)
        print('faor_converted is,', faro_converted)
        print('ref is', ref)
        angular_error_tem = np.inner(ref, axis)
        angular_error_tem = angular_error_tem /(np.linalg.norm(ref) * np.linalg.norm(axis))
        angular_error_tem = np.arccos(angular_error_tem) * 180/np.pi
        if angular_error_tem > 90:
            angular_error_tem = 180 - angular_error_tem
        angular_error.append(angular_error_tem)
    return np.asarray(angular_error)


# Function to calculate errors and generate plots
# Use the transformed raw and faro data access the accuracy
# Input:
#           pos: IOS cylinder base
#           angle: IOS cylinder axis
#           faro: Faro base
# Output:
#           position error in plane.
#           height error (w.r.t 1st cylinder)
#           angular error of axis
def estimate_full_relative(pos, angle, faro):
    pos_error = np.sqrt(np.sum((pos[0:7,0:2] - faro[0:7,0:2])**2, axis=1))
    height_error = pos[:,2] - pos[0,2]
    angular_error = check_cylinder_orientation_full_relative(angle)
    print('position error is', pos_error)
    print('height error is', height_error)
    print('angular error is', angular_error)
    fig = plt.figure()
    plt.scatter(faro[:, 0], faro[:, 1], color = 'black', label = 'faro')
    plt.scatter(pos[:, 0], pos[:, 1], color='red', label = 'IOS')
    for j in range(7):
        plt.text(faro[j,0]-2, faro[j,1] + 0.8, np.str(round(angular_error[j],2)) + ' degree', color = 'blue')
        plt.text(faro[j, 0] - 2, faro[j, 1] + 2.5, np.str(round(pos_error[j], 2)) + ' mm', color='blue')
        plt.arrow(pos[j, 0], pos[j, 1], angle[j, 0] * 40, angle[j, 1] * 40,
                  color='purple', width=0.13)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.legend()
    plt.xlim(-40,5)
    plt.ylim(-5, 40)
    plt.title('IOS full arch accuracy estimation (relative angular)')
    return pos_error, height_error, angular_error



# Function to calculate errors and generate plots
# Use the transformed raw and faro data access the accuracy
# Input:
#           pos: IOS cylinder base
#           angle: IOS cylinder axis
#           faro: Faro base
# Output:
#           position error in plane.
#           height error (w.r.t 1st cylinder)
#           angular error of axis
def estimate_full_absolute(pos, angle, faro, faro_axis):
    pos_error = np.sqrt(np.sum((pos[0:7,0:2] - faro[0:7,0:2])**2, axis=1))
    height_error = pos[:,2] - pos[0,2]
    angular_error = check_cylinder_orientation_full_absolute(angle, faro_axis)
    print('position error is', pos_error)
    print('height error is', height_error)
    print('angular error is', angular_error)
    fig = plt.figure()
    plt.scatter(faro[:, 0], faro[:, 1], color = 'black', label = 'faro')
    plt.scatter(pos[:, 0], pos[:, 1], color='red', label = 'IOS')
    for j in range(7):
        plt.text(faro[j,0]-2, faro[j,1] + 0.8, np.str(round(angular_error[j],2)) + ' degree', color = 'blue')
        plt.text(faro[j, 0] - 2, faro[j, 1] + 2.5, np.str(round(pos_error[j], 2)) + ' mm', color='blue')
        plt.arrow(pos[j, 0], pos[j, 1], (angle[j, 0] - faro_axis[j,0]) * 40, (angle[j, 1] - faro_axis[j,1]) * 40,
                  color='purple', width=0.13)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.legend()
    plt.xlim(-40,5)
    plt.ylim(-5, 40)
    plt.title('IOS full arch accuracy estimation (absolute angular)')
    return pos_error, height_error, angular_error



# Flip vector based on axis direction, making sure vector is pointing to positive direction
# Input:
#       vector
#       axis: which axis to flip ('x', 'y', 'z')
def flip_vector(vector, axis = 'z'):
    tem = np.copy(vector)
    if axis == 'x':
        if tem[0] < 0:
            tem = -tem
    if axis == 'y':
        if tem[1] < 0:
            tem = -tem
    if axis == 'z':
        if tem[2] < 0:
            tem = -tem
    return tem

if __name__ == '__main__':
    #file_path = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\Results\\Accuracy FXT tests\\" \
    #            "cylinder_measurements.txt"
    base_path = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\Results\\Accuracy FXT tests\\"
    file_path = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\Results\\Accuracy FXT tests\\" \
                "cylinder_plane_measurements.txt"
    faro_file_path = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\Results\\Accuracy FXT tests\\" \
                "full_arch_faro.txt"


    accuracy_raw = Yomiread.read_csv(file_path,6,20)
    faro_raw = Yomiread.read_csv(faro_file_path,6,20)
    #trial1_raw = accuracy_raw[0:5,:]
    #trial2_raw = accuracy_raw[5:10,:]
    #print('raw data is', accuracy_raw)
    #print('trial2_raw is', trial2_raw)

    #raw = trial1_raw
    #cylinder_base = check_cylinder_base(raw, 5, 3)
    #cylinder_angular_error = check_cylinder_orientation(raw)
    #cylinder_angular_error2 = check_cylinder_orientation(trial2_raw)

    #print('cylinder_base is', cylinder_base)
    #print('cylinder_orientation is', cylinder_angular_error)
    #print('cylinder_orientation is', cylinder_angular_error2)

    cylinder_base, cylinder_axis = check_cylinder_base_full(accuracy_raw, 1, 7, reference='cylinder')
    faro_base, faro_axis = check_cylinder_base_full(faro_raw, 1, 7, axis='y', reference='cylinder')

    pos_error, height_error, angular_error = estimate_full_relative(cylinder_base, cylinder_axis, faro_base)
    pos_error2, height_error2, angular_error2 = estimate_full_absolute(cylinder_base, cylinder_axis, faro_base, faro_axis)
    Yomiwrite.write_csv_array(base_path+'pos_error.txt', pos_error)
    Yomiwrite.write_csv_array(base_path + 'height_error.txt', height_error)
    Yomiwrite.write_csv_array(base_path+'angular_error.txt', angular_error)


    # pos_error = np.sqrt(np.sum((cylinder_base[0:7,0:2] - faro_base[0:7,0:2])**2, axis=1))
    # height_error = cylinder_base[:,2] - cylinder_base[0,2]
    # angular_error = check_cylinder_orientation_full(cylinder_axis)
    # print('position error is', pos_error)
    # print('height error is', height_error)
    # print('angular error is', angular_error)
    #
    # fig = plt.figure()
    # plt.scatter(faro_base[:, 0], faro_base[:, 1], color = 'black', label = 'faro')
    # plt.scatter(cylinder_base[:, 0], cylinder_base[:, 1], color='red', label = 'IOS')
    # for j in range(7):
    #     plt.text(faro_base[j,0]-2, faro_base[j,1] + 0.8, np.str(round(angular_error[j],2)) + ' degree', color = 'blue')
    #     plt.text(faro_base[j, 0] - 2, faro_base[j, 1] + 2.5, np.str(round(pos_error[j], 2)) + ' mm', color='blue')
    #     plt.arrow(cylinder_base[j, 0], cylinder_base[j, 1], cylinder_axis[j, 0] * 40, cylinder_axis[j, 1] * 40,
    #               color='purple', width=0.13)
    # plt.xlabel('x (mm)')
    # plt.ylabel('y (mm)')
    # plt.legend()
    # plt.xlim(-40,5)
    # plt.ylim(-5, 40)
    # plt.title('IOS full arch accuracy estimation')
    plt.show()