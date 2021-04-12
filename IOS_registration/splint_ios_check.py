import numpy as np
import Readers as Yomiread
import local_registration
import affine_registration
import Kinematics as Yomikin
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import Axes3D

def transpose_pc(pc_2b_convert, transformation):
    pc_converted = []
    for point in pc_2b_convert:
        tem = np.insert(point, 3, 1.)
        tem_converted = np.matmul(transformation, tem)[0:3]
        pc_converted.append(tem_converted)
    return np.asarray(pc_converted)


trial1_file = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\Trial1 - FXT1\\peak_location.txt"
trial2_file = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\Trial2\\peak_location.txt"
ground_file = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\IOS_scan_raw_data\\IOS Splint\\designed_landmarks.txt"
trial1 = Yomiread.read_csv(trial1_file,4)[:,1:]
trial2 = Yomiread.read_csv(trial2_file,4)[:,1:]
ground = Yomiread.read_csv(ground_file,4)[:,1:]

trans_init = np.eye(4)
rigid_init_parameter = Yomikin.Yomi_parameters(trans_init)
affine_rigid_part = affine_registration.rigid_registration(rigid_init_parameter, ground, trial2)
trans_rigid = Yomikin.Yomi_Base_Matrix(affine_rigid_part)

trial1_transformed = transpose_pc(trial1, trans_rigid)
print('delta is', trial1_transformed - ground)

fig2 = plt.figure(2)
ax3d = fig2.add_subplot(111, projection='3d')
# #ax3d.plot(x_knots, y_knots, z_knots, color = 'blue', label = 'knots')
# ax3d.plot(spline_target[:,0], spline_target[:,1], spline_target[:,2], color = 'r', label = 'target spline')
# ax3d.plot(spline_target[:, 0], spline_target[:, 1], spline_target[:, 2], 'r*')
# ax3d.plot(spline_rigid[:, 0], spline_rigid[:, 1], spline_rigid[:, 2], color = 'b', label='rigid spline')
# ax3d.plot(spline_rigid_moved[:, 0], spline_rigid_moved[:, 1], spline_rigid_moved[:, 2], color='g', label='moved spline')
ax3d.plot(ground[:,0], ground[:,1], ground[:,2], 'b', label='ground')
#ax3d.plot(trial1[:,0], trial1[:,1], trial1[:,2], color='r', label='trial1')
ax3d.plot(trial1_transformed[:,0], trial1_transformed[:,1], trial1_transformed[:,2], color='r', label='trial1')
# #ax3d.plot(x_fine_d, y_fine_d, z_fine_d, color='black', label='fine_d')
fig2.show()
plt.legend()
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
