import numpy as np
import coordinates
import Readers as Yomiread
from tkinter import filedialog

def calculate_base_point(cylinder_middle_point, drill_axis, drill_bottom_point):
    M = cylinder_middle_point
    B = drill_bottom_point
    A = drill_axis
    mtx = np.array([[A[0], A[1], A[2], 0],
                    [1, 0, 0, -A[0]],
                    [0, 1, 0, -A[1]],
                    [0, 0, 1, -A[2]]])
    RHS = np.array([A[0]*B[0]+A[1]*B[1]+A[2]*B[2], M[0], M[1], M[2]])
    result = np.matmul(np.linalg.inv(mtx), RHS)
    drill_center_bottom_point = result[0:3]
    return drill_center_bottom_point


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


#file_path = filedialog.askopenfilename()
base_path = filedialog.askdirectory()
faro_file = base_path + '\\faro_measurement.csv'
implant_file = base_path + '\\implant_transforamtion_in_fiducial_space.csv'
IMPLANT_LENGTH = 8.5

GOLDEN_T = np.array([[-0.000185716, 0.999999, -0.00122936, -12.702999999999999],
                                  [-0.999999, -0.000184434, 0.00104337, 4.058],
                                  [0.00104315, 0.00122956, 0.999999, -7.938],
                                  [0, 0, 0, 1]])

# read faro measurement
data = Yomiread.read_faro_ios_implant_measurement(faro_file)

print('data is', data)
sphere = data[1]
plane = data[3]
point = data[5]
cylinder = data[7]
pin = data[9]

sphere1 = np.asarray(sphere[0][0:3])
sphere2 = np.asarray(sphere[1][0:3])
sphere3 = np.asarray(sphere[2][0:3])
fiducial_frame = coordinates.generate_frame_yomi(sphere1, sphere2, sphere3)
mtx = np.matmul(GOLDEN_T, fiducial_frame)

cylinder_mid_point = np.asarray(cylinder[0][0:3])
cylinder_axis = np.asarray(cylinder[0][3:6])
pin_length = np.asarray(pin[0][0])

plane_point = np.asarray(plane[0][0:3])
plane_axis = np.asarray(plane[0][3:6])

point_location = np.asarray(point[0][0:3])

drill_axis = -cylinder_axis # along the drill axis (pointing down)
drill_mid_point = cylinder_mid_point
drill_top_point = point_location
drill_bottom_point = drill_top_point + pin_length * drill_axis

base_center = calculate_base_point(drill_mid_point, drill_axis, drill_bottom_point)
implant_position = base_center - IMPLANT_LENGTH/2 * drill_axis

actual_implant_position = transpose_pc(implant_position, mtx)
actual_axis = np.matmul(mtx[0:3,0:3], drill_axis)

actual_base_center = transpose_pc(base_center, mtx)
actual_implant_position_check = actual_base_center - IMPLANT_LENGTH/2 * actual_axis

# read designed implant
design_implant = Yomiread.read_csv(implant_file,4)
print(design_implant)

actual_implant_position_new = np.asarray([-9.459, -60.397, -46.437])
actual_axis_new = np.asarray([-0.1302, 0.6442, -0.7537])

#implant_point_difference = actual_implant_position - design_implant[0:3,3]
implant_point_difference = actual_implant_position - design_implant[0:3,3]

theta = np.arccos(np.inner(implant_point_difference, actual_axis)/(np.linalg.norm(implant_point_difference)*np.linalg.norm(actual_axis)))

depth_error = np.cos(theta) * np.linalg.norm(implant_point_difference)
lateral_error = np.sin(theta) * np.linalg.norm(implant_point_difference)


#depth_error = np.matmul(implant_point_difference, actual_axis)
#print('depth_error is', depth_error)
#print('actual_base_center is', actual_base_center)
#print('base center is', base_center)
#print('implant difference norm is', np.linalg.norm(depth_error))
print('actual_implant_position is', actual_implant_position)
print('actual_drill axis is', actual_axis)

#print('actual_axis is', actual_axis)
angular_error = np.arccos(np.matmul(actual_axis, design_implant[0:3,0]) / (np.linalg.norm(actual_axis) * np.linalg.norm(design_implant[0:3,0]))) * 180/np.pi
print('angular_error is', angular_error)
print('depth_error is', depth_error)
print('lateral_error is', lateral_error)