import numpy as np
import Readers as Yomiread
import open3d as o3d


def combine_list_elements(list):
    total = np.ones((1,3))
    for item in list:
        total = np.vstack((total, item))
    return total[1:,:]


def find_ctl_points(source, target, transformation):
    # Use target points (dicom) as control points in both images
    # https://github.com/intel-isl/Open3D/issues/2368

    target_down = down_sampling_pc(target)
    source_down = prepare_ctl_points(target_down, transformation)

    target_pc = o3d.PointCloud()
    target_pc.points = o3d.Vector3dVector(target_down)
    target_mean, target_matrix = o3d.compute_point_cloud_mean_and_covariance(target_pc)
    target_axis = np.linalg.eig(target_matrix)
    target_ctl = [target_down, target_mean, target_axis[1]]

    source_pc = o3d.PointCloud()
    source_pc.points = o3d.Vector3dVector(source_down)
    source_mean, source_matrix = o3d.compute_point_cloud_mean_and_covariance(source_pc)
    source_axis = np.linalg.eig(source_matrix)
    source_ctl = [source_down, source_mean, source_axis[1]]
    return source_ctl, target_ctl


def down_sampling_pc(pc):
    # print('shape of original pc is', np.shape(pc))
    down_pcd = o3d.geometry.PointCloud()
    down_pcd.points = o3d.Vector3dVector(pc)
    # pc_ds = o3d.geometry.voxel_down_sample(down_pcd, 2)
    pc_ds = o3d.geometry.voxel_down_sample(down_pcd, 2)
    pc_ds_points = pc_ds.points
    # print('shape of downsample pc is', np.shape(pc_ds_points))
    return pc_ds_points
    # return pc


def prepare_ctl_points(target, transformation):
    target_in_source = []
    for point in target:
        tem_t = np.insert(point, 3, 1)
        extra_source_tem = np.matmul(np.linalg.inv(transformation), tem_t)
        extra_source_tem = extra_source_tem[0:3]
        target_in_source.append(extra_source_tem)
    target_in_source = np.asarray(target_in_source)
    return target_in_source


# Read source and target raw data
source_file_R1 = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Rmolar1.csv'
source_file_R2 = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Rmolar2.csv'
source_file_R3 = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Rmolar3.csv'
source_file_L1 = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Lmolar1.csv'
source_file_L2 = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Lmolar2.csv'
source_file_L3 = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Lmolar3.csv'
source_file_full = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_full.csv'
source_file_incisor = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_incisor.csv'

target_file_R1 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Rmolar1.csv'
target_file_R2 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Rmolar2.csv'
target_file_R3 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Rmolar3.csv'
target_file_L1 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Lmolar1.csv'
target_file_L2 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Lmolar2.csv'
target_file_L3 = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_Lmolar3.csv'
target_file_cylinder = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_full.csv'
target_file_incisor = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_incisor.csv'
target_file_check_cylinder = 'G:\My Drive\Project\IntraOral Scanner Registration\stl_points_cylinder.csv'

accuracy_file = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\Results\\Accuracy FXT tests\\Accuracy assessment\\stl_measurements\\full_arch_cylinder_measurement_5.txt"
accuracy_raw = Yomiread.read_csv(accuracy_file, 6, 20)

source_points_R1 = Yomiread.read_csv(source_file_R1, 3, -1)
source_points_R2 = Yomiread.read_csv(source_file_R2, 3, -1)
source_points_R3 = Yomiread.read_csv(source_file_R3, 3, -1)
source_points_L1 = Yomiread.read_csv(source_file_L1, 3, -1)
source_points_L2 = Yomiread.read_csv(source_file_L2, 3, -1)
source_points_L3 = Yomiread.read_csv(source_file_L3, 3, -1)
source_points_full = Yomiread.read_csv(source_file_full, 3, -1)
source_points_incisor = Yomiread.read_csv(source_file_incisor, 3, -1)

target_points_R1 = Yomiread.read_csv(target_file_R1, 3, -1)
target_points_R2 = Yomiread.read_csv(target_file_R2, 3, -1)
target_points_R3 = Yomiread.read_csv(target_file_R3, 3, -1)
target_points_L1 = Yomiread.read_csv(target_file_L1, 3, -1)
target_points_L2 = Yomiread.read_csv(target_file_L2, 3, -1)
target_points_L3 = Yomiread.read_csv(target_file_L3, 3, -1)
target_points_cylinder = Yomiread.read_csv(target_file_cylinder, 3, -1)
target_points_incisor = Yomiread.read_csv(target_file_incisor, 3, -1)
target_points_check_cylinder = Yomiread.read_csv(target_file_check_cylinder, 3, -1)

# dicom_points = np.vstack((source_points_R1, source_points_R2))
# dicom_points = np.vstack((dicom_points, source_points_R3))
# dicom_points = np.vstack((dicom_points, source_points_L1))
# dicom_points = np.vstack((dicom_points, source_points_L2))
# dicom_points = np.vstack((dicom_points, source_points_L3))
# dicom_points = np.vstack((dicom_points, source_points_full))
# dicom_points = np.vstack((dicom_points, source_points_incisor))
#
# stl_points = np.vstack((target_points_R1, target_points_R2))
# stl_points = np.vstack((stl_points, target_points_R3))
# stl_points = np.vstack((stl_points, target_points_L1))
# stl_points = np.vstack((stl_points, target_points_L2))
# stl_points = np.vstack((stl_points, target_points_L3))
# stl_points = np.vstack((stl_points, target_points_cylinder))
# stl_points = np.vstack((stl_points, target_points_incisor))
# stl_points = np.vstack((stl_points, target_points_check_cylinder))

dicom = []
stl = []

dicom.append(source_points_R1)
dicom.append(source_points_R2)
dicom.append(source_points_R3)
dicom.append(source_points_L1)
dicom.append(source_points_L2)
dicom.append(source_points_L3)
dicom.append(source_points_incisor)
dicom.append(source_points_full)
dicom_points = combine_list_elements(dicom)

stl.append(target_points_R1)
stl.append(target_points_R2)
stl.append(target_points_R3)
stl.append(target_points_L1)
stl.append(target_points_L2)
stl.append(target_points_L3)
stl.append(target_points_incisor)
stl.append(target_points_cylinder)
stl.append(target_points_check_cylinder)
stl_points = combine_list_elements(stl)

