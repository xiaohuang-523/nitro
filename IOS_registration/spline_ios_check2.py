import numpy as np
import Readers as Yomiread
import local_registration
import affine_registration
import Kinematics as Yomikin
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import Axes3D
import Writers as Yomiwrite
from scipy import interpolate
import coordinates
import spline_correction as sc
import feature_extraction as fe


def transpose_pc(pc_2b_convert, transformation):
    pc_converted = []
    for point in pc_2b_convert:
        tem = np.insert(point, 3, 1.)
        tem_converted = np.matmul(transformation, tem)[0:3]
        pc_converted.append(tem_converted)
    return np.asarray(pc_converted)


trial1_file = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\Trial1 - FXT1\\peak_location.txt"
trial2_file = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\Trial2\\peak_location.txt"
trial3_file = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\Attached_full\\4152021-Full_IOS_Splint-4152021\\peak_location.txt"
trial3_result = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\Attached_full\\4152021-Full_IOS_Splint-4152021\\transformed_ground.txt"

v2_file = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\IOS_splint_v2\\attached_trial1\\trial1\\peak_location.txt"
v2_result_before = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\IOS_splint_v2\\attached_trial1\\trial1\\rigid_error.txt"
v2_result_after = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\IOS_splint_v2\\attached_trial1\\trial1\\non-rigid_error.txt"

v2_file_2 = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\IOS_splint_v2\\separate\\trial1\\peak_location.txt"
v2_result_2_before = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\IOS_splint_v2\\separate\\trial1\\rigid_error.txt"
v2_result_2_after = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\IOS_splint_v2\\separate\\trial1\\non-rigid_error.txt"

ground_file = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\designed_landmarks.txt"
ground_v2_file = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\designed_landmarks_v2.txt"
#trial1 = Yomiread.read_csv(trial1_file, 4)[:,1:]
#trial2 = Yomiread.read_csv(trial2_file, 4)[:,1:]

trial3 = Yomiread.read_csv(v2_file_2, 4)[:,1:]
ground = Yomiread.read_csv(ground_v2_file,4)[:,1:]

trans_init = np.eye(4)
rigid_init_parameter = Yomikin.Yomi_parameters(trans_init)
affine_rigid_part = affine_registration.rigid_registration(rigid_init_parameter, ground, trial3)
trans_rigid = Yomikin.Yomi_Base_Matrix(affine_rigid_part)

modify_parameters = np.array([0, 0, 0, -np.pi/6, 0, 0])
modify_matrix = Yomikin.Yomi_Base_Matrix(modify_parameters)


ground_transformed = transpose_pc(ground, np.linalg.inv(trans_rigid))

#trial3 = transpose_pc(trial3, modify_matrix)
#ground_transformed = transpose_pc(ground_transformed, modify_matrix)

delta = trial3 - ground_transformed
print('delta is', trial3 - ground_transformed)
print('delta is', np.linalg.norm(delta, axis=1))

center_2 = np.array([-30.3, 6.5, 211])

#Yomiwrite.write_csv_matrix(v2_result, ground_transformed)

#fiducial_list = np.array([0, 1, 4, 7])
#fiducial_list = np.asarray(range(8))
fiducial_list = np.array([0, 1, 2, 3, 4, 5, 6, 7])

splint_geometry = fe.Splint(fiducial_list, type='Geometry', spline_base_axis='x')
splint_ios = fe.Splint(fiducial_list, type='IOS', spline_base_axis='x')
splint_ios_corrected = fe.Splint(fiducial_list, type='IOS', spline_base_axis='x')

#ground_transformed = np.flip(ground_transformed, 0)
#trial3 = np.flip(trial3, 0)

#ground_transformed_tem = np.copy(ground_transformed)
#trial3_tem = np.copy(trial3)
#print('ground_transformed_tem is', ground_transformed)

#ground_transformed[:,0] = ground_transformed_tem[:,1]
#ground_transformed[:,1] = ground_transformed_tem[:,0]
#trial3[:,0] = trial3_tem[:,1]
#trial3[:,1] = trial3_tem[:,0]
#print('ground_transformed_tem is', ground_transformed)

for i in range(np.shape(trial3)[0]):
    splint_geometry.add_pyramid(i, ground_transformed[i,:])
    splint_ios.add_pyramid(i, trial3[i,:])

print('fiducial list is', splint_geometry.pyramid_fiducial_list)
print('target list is', splint_geometry.pyramid_target_list)
print('all list is', splint_geometry.pyramid_number_list)

splint_geometry.update_spline(fine_flag=True)
splint_ios.update_spline(fine_flag=True)

#print('splint_fine points are', splint_geometry.spline_points_fine)
#print('spline boundary is', splint_geometry.spline_points_fine_cylindrical_mid_points)

print('ground transformed ')
displacement = np.asarray(splint_geometry.spline_points_fine) - np.asarray(splint_ios.spline_points_fine)

for i in splint_ios.pyramid_number_list:
    point = splint_ios.get_pyramid(i)
    print('i is', i)
    print('point is', point)
    point_cylindrical = coordinates.convert_cylindrical(point, splint_ios.spline_points_cylindrical_center)
    #point_cylindrical = coordinates.convert_cylindrical(point, center_2)
    point_moved = sc.displacement_single_point(point, point_cylindrical, splint_ios.spline_points_fine_cylindrical_mid_points, displacement)
    print('moved point is', point_moved)
    print('target point is', splint_geometry.get_pyramid(i))
    splint_ios_corrected.add_pyramid(i, point_moved)

diff_before = np.asarray(splint_geometry.pyramid_points) - np.asarray(splint_ios.pyramid_points)
diff_after = np.asarray(splint_geometry.pyramid_points) - np.asarray(splint_ios_corrected.pyramid_points)
diff_before_norm = np.linalg.norm(diff_before, axis=1)
diff_after_norm = np.linalg.norm(diff_after, axis=1)
diff_before_norm_2d = np.linalg.norm(diff_before[:,0:2], axis=1)
diff_after_norm_2d = np.linalg.norm(diff_after[:,0:2], axis=1)

Yomiwrite.write_csv_matrix(v2_result_2_before, diff_before_norm)
Yomiwrite.write_csv_matrix(v2_result_2_after, diff_after_norm)

print('point is', splint_geometry.pyramid_points)

fig1 = plt.figure()
plt.scatter(range(np.shape(splint_geometry.spline_points_fine_cylindrical_mid_points)[0]), splint_geometry.spline_points_fine_cylindrical_mid_points, label='theta check')
#plt.scatter(range(np.shape(splint_geometry.spline_points_fine_cylindrical_mid_points)[0]), splint_geometry.spline_points_fine_cylindrical_mid_points[:,0], label='r check')
#plt.scatter(range(np.shape(splint_geometry.spline_points_fine)[0]), splint_geometry.spline_points_fine[:,1], label='y check')
#plt.scatter(range(np.shape(splint_geometry.spline_points_fine)[0]), splint_geometry.spline_points_fine[:,0], label='x check')
plt.legend()

fig2 = plt.figure(2)
ax3d = fig2.add_subplot(111, projection='3d')
# #ax3d.plot(x_knots, y_knots, z_knots, color = 'blue', label = 'knots')
# ax3d.plot(spline_target[:,0], spline_target[:,1], spline_target[:,2], color = 'r', label = 'target spline')
# ax3d.plot(spline_target[:, 0], spline_target[:, 1], spline_target[:, 2], 'r*')
# ax3d.plot(spline_rigid[:, 0], spline_rigid[:, 1], spline_rigid[:, 2], color = 'b', label='rigid spline')
# ax3d.plot(spline_rigid_moved[:, 0], spline_rigid_moved[:, 1], spline_rigid_moved[:, 2], color='g', label='moved spline')
ax3d.plot(np.asarray(splint_geometry.pyramid_points)[:,0], np.asarray(splint_geometry.pyramid_points)[:,1], np.asarray(splint_geometry.pyramid_points)[:,2], 'b', label='ground')
#ax3d.plot(trial1[:,0], trial1[:,1], trial1[:,2], color='r', label='trial1')
ax3d.plot(np.asarray(splint_ios.pyramid_points)[:,0], np.asarray(splint_ios.pyramid_points)[:,1], np.asarray(splint_ios.pyramid_points)[:,2], color='r', label='trial3 original')
ax3d.plot(np.asarray(splint_ios_corrected.pyramid_points)[:,0], np.asarray(splint_ios_corrected.pyramid_points)[:,1], np.asarray(splint_ios_corrected.pyramid_points)[:,2], color='g', label='trial3 corrected')
ax3d.scatter(np.asarray(splint_ios_corrected.pyramid_fiducial_points)[:,0], np.asarray(splint_ios_corrected.pyramid_fiducial_points)[:,1], np.asarray(splint_ios_corrected.pyramid_fiducial_points)[:,2], color='g', label='trial3 corrected fiducial')
ax3d.set_xlabel('X(mm)')
ax3d.set_ylabel('Y(mm)')
ax3d.set_zlabel('Z(mm)')
fig2.show()
plt.legend()

fig3 = plt.figure()
plt.scatter(range(len(diff_before_norm)), diff_before_norm, label='before')
plt.scatter(range(len(diff_after_norm)), diff_after_norm, label='after')
plt.legend()

fig6 = plt.figure()
plt.scatter(range(len(diff_before_norm_2d)), diff_before_norm_2d, label='2d before')
plt.scatter(range(len(diff_after_norm_2d)), diff_after_norm_2d, label='2d after')
plt.legend()

fig5 = plt.figure()
plt.scatter(range(np.shape(np.asarray(splint_geometry.pyramid_points))[0]), np.asarray(splint_geometry.pyramid_points)[:,0] - np.asarray(splint_ios_corrected.pyramid_points)[:,0], label='error x')
plt.scatter(range(np.shape(np.asarray(splint_geometry.pyramid_points))[0]), np.asarray(splint_geometry.pyramid_points)[:,1] - np.asarray(splint_ios_corrected.pyramid_points)[:,1], label='error y')
plt.scatter(range(np.shape(np.asarray(splint_geometry.pyramid_points))[0]), np.asarray(splint_geometry.pyramid_points)[:,2] - np.asarray(splint_ios_corrected.pyramid_points)[:,2], label='error z')
plt.legend()

plt.show()

exit()





ground_new = Yomiread.read_csv(trial3_result,3)
ground_new[:,0] = ground_new[:,0]
ground_del = np.delete(ground_new, 6, 0)
ground_del = np.delete(ground_del, 5, 0)
ground_del = np.delete(ground_del, 3, 0)
ground_del = np.delete(ground_del, 2, 0)

# x = ground_del[:,0]
# y = ground_del[:,1]
# z = ground_del[:,2]

ios_spline = trial3
ios_spline = np.delete(ios_spline, 7, 0)
ios_spline = np.delete(ios_spline, 5, 0)
#ios_spline = np.delete(ios_spline, 4, 0)
ios_spline = np.delete(ios_spline, 3, 0)
ios_spline = np.delete(ios_spline, 2, 0)


x = ios_spline[:,0]
y = ios_spline[:,1]
z = ios_spline[:,2]

#print('ground_del is', ground_del)
print('ios spline is', ios_spline)

#f = interpolate.interp1d(x, y, kind='cubic', fill_value="extrapolate")
#x_new = np.arange(np.min(x), np.max(x), 0.01)
#y_new = f(x_new)

fyx = interpolate.interp1d(y, x, kind='cubic', fill_value="extrapolate")
y_new = np.arange(np.min(y)-5, np.max(y) + 5, 0.01)
x_new = fyx(y_new)
fyz = interpolate.interp1d(y, z, kind='cubic', fill_value="extrapolate")
z_new = fyz(y_new)








fig2 = plt.figure(2)
ax3d = fig2.add_subplot(111, projection='3d')
# #ax3d.plot(x_knots, y_knots, z_knots, color = 'blue', label = 'knots')
# ax3d.plot(spline_target[:,0], spline_target[:,1], spline_target[:,2], color = 'r', label = 'target spline')
# ax3d.plot(spline_target[:, 0], spline_target[:, 1], spline_target[:, 2], 'r*')
# ax3d.plot(spline_rigid[:, 0], spline_rigid[:, 1], spline_rigid[:, 2], color = 'b', label='rigid spline')
# ax3d.plot(spline_rigid_moved[:, 0], spline_rigid_moved[:, 1], spline_rigid_moved[:, 2], color='g', label='moved spline')
ax3d.plot(ground_transformed[:,0], ground_transformed[:,1], ground_transformed[:,2], 'b', label='ground')
#ax3d.plot(trial1[:,0], trial1[:,1], trial1[:,2], color='r', label='trial1')
ax3d.plot(trial3[:,0], trial3[:,1], trial3[:,2], color='r', label='trial3')
ax3d.scatter(x, y, z, color='black', label='guided points')
ax3d.set_xlabel('X(mm)')
ax3d.set_ylabel('Y(mm)')
ax3d.set_zlabel('Z(mm)')
fig2.show()
plt.legend()







fig1 = plt.figure()
plt.scatter(x, y, color = 'red')
plt.plot(x_new, y_new, color='green')
#plt.xlim([-80, 10])

fig3 = plt.figure()
plt.scatter(y, z, color = 'red')
plt.plot(y_new, z_new, color = 'green')

plt.show()

exit()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z, color='blue')
#ax.scatter(ground[:,0], ground[:,1], ground[:,2], color='red')
plt.show()






exit()
rms_init = 1
count = 1
voxel_step = 0.02
voxel_size_molar = 0.1
THRESHOLD_ICP = 50
icp1 = local_registration.registration(voxel_size_molar, THRESHOLD_ICP , trial1, ground, np.eye(4))
icp2 = local_registration.registration(voxel_size_molar, THRESHOLD_ICP , trial1, ground, icp1.transformation)
icp3 = local_registration.registration(voxel_size_molar, THRESHOLD_ICP , trial1, ground, icp2.transformation)
print('transformation is', icp3.transformation)
print('icp is', icp3)
