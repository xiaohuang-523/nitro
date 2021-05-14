import numpy as np
import Readers as Yomiread
import coordinates
import geometry

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

# faro data
faro_measurement = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\TRE_verification\\faro_measurement\\faro_measurement.csv"
faro = Yomiread.read_faro_ios_splint_measurement(faro_measurement)
print('faro is', faro)
print('plane is', faro[3])
print('plane label is', faro[4])

p1_faro = np.asarray(faro[1][0][0:3])
p2_faro = np.asarray(faro[1][1][0:3])
p3_faro = np.asarray(faro[1][2][0:3])

plane1 = geometry.plane3D(faro[3][0][0:3], faro[3][0][3:6])
plane2 = geometry.plane3D(faro[3][1][0:3], faro[3][1][3:6])
plane3 = geometry.plane3D(faro[3][2][0:3], faro[3][2][3:6])
plane4 = geometry.plane3D(faro[3][3][0:3], faro[3][3][3:6])
plane5 = geometry.plane3D(faro[3][4][0:3], faro[3][4][3:6])

t1_faro = geometry.plane_intersection(plane1, plane2, plane3)
t2_faro = geometry.plane_intersection(plane2, plane3, plane4)
t3_faro = geometry.plane_intersection(plane3, plane4, plane5)

frame_faro = coordinates.generate_frame(p1_faro, p2_faro, p3_faro)
t1_faro_real = transpose_pc(t1_faro, frame_faro)
t2_faro_real = transpose_pc(t2_faro, frame_faro)
t3_faro_real = transpose_pc(t3_faro, frame_faro)
target_frame_faro = coordinates.generate_frame(t1_faro_real, t2_faro_real, t3_faro_real)
z_faro = np.linalg.inv(target_frame_faro)[0:3, 2]

# ios measurement
ios_measurement = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\TRE_verification\\result\\"
target_file = "target_points.txt"
frame_file = "frame_points.txt"
ios_target = Yomiread.read_csv(ios_measurement + target_file, 3)
ios_frame = Yomiread.read_csv(ios_measurement + frame_file, 3)

frame_ios = coordinates.generate_frame(ios_frame[0,:], ios_frame[1,:], ios_frame[2,:])
t1_ios_real = transpose_pc(ios_target[0,:], frame_ios)
t2_ios_real = transpose_pc(ios_target[1,:], frame_ios)
t3_ios_real = transpose_pc(ios_target[2,:], frame_ios)
target_frame_ios = coordinates.generate_frame(t1_ios_real, t2_ios_real, t3_ios_real)
z_ios = np.linalg.inv(target_frame_ios)[0:3, 2]

angle_error = np.arccos(np.matmul(z_faro, z_ios)/(np.linalg.norm(z_faro) * np.linalg.norm(z_ios))) * 180 / np.pi



# check results
print('t1_faro is', t1_faro_real)
print('t1_iso is', t1_ios_real)
print('dis1 is', np.linalg.norm(t1_ios_real - t1_faro_real))
print('dis2 is', np.linalg.norm(t2_ios_real - t2_faro_real))
print('dis3 is', np.linalg.norm(t3_ios_real - t3_faro_real))
print('dis1 is', t1_ios_real - t1_faro_real)
print('dis2 is', t2_ios_real - t2_faro_real)
print('dis3 is', t3_ios_real - t3_faro_real)
print('angular error is', angle_error)