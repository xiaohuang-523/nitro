# Prepare features for non-rigid registration
# By Xiao Huang
# 04/05/2021

import open3d as o3d
import numpy as np
import Readers as Yomiread
from scipy import interpolate


def combine_pc(list):
    combined_pc = list[0]
    for i in range(len(list)-1):
        #combined_pc = np.vstack((combined_pc, list[i+1]))
        combined_pc = np.concatenate((combined_pc,list[i+1]))
    return combined_pc


# Convert to cylindrical coordinates
def convert_cylindrical(pc, center):
    theta = []
    r = []
    z = []
    for point in pc:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        angle = np.arctan2(dy, dx)
        #if angle < 0:
        #    angle = angle + 2*np.pi
        theta.append(angle)
        r.append(np.sqrt(dx**2 + dy**2))
        z.append(point[2])
    pc_cylindrical = np.asarray([r, theta, z])
    return pc_cylindrical.transpose()


def arc_length(x, y):
    arc = np.sqrt((x[1] - x[0])**2 + (y[1]-y[0])**2)
    for i in range(len(x)-2):
        arc = arc + np.sqrt((x[i+2] - x[i+1])**2 + (y[i+2]-y[i+1])**2)
    return arc


def arc_segment(x, y, arc_total_length, segment_number):
    arc = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
    count = 1
    segment_points_x = []
    segment_points_y = []
    for i in range(len(x) - 2):
        arc = arc + np.sqrt((x[i + 2] - x[i + 1]) ** 2 + (y[i + 2] - y[i + 1]) ** 2)
        if arc > count * arc_total_length / segment_number:
            count += 1
            segment_points_x.append(x[i])
            segment_points_y.append(y[i])
    segment_points = np.asarray([segment_points_x, segment_points_y]).transpose()

    return segment_points


def fit_spline_and_split(spline_points, segment_number):
    x = spline_points[:,0]
    y = spline_points[:,1]
    z = spline_points[:,2]

    tck_xy = interpolate.splrep(x, y, s=0)
    tck_xz = interpolate.splrep(x, z, s=0)
    deltax = 1e-5
    spline_fine_point = []
    for j in range(len(x)-1):
        print('refine spline for part', j)
        start = x[j]
        end = x[j+1]
        segment_points = np.zeros((segment_number,3))
        x_tem = np.arange(start, end, deltax)
        y_tem = interpolate.splev(x_tem, tck_xy, der=0)
        z_tem = interpolate.splev(x_tem, tck_xz, der=0)
        length_xy = arc_length(x_tem, y_tem)
        length_xz = arc_length(x_tem, z_tem)

        segments_points_xy = arc_segment(x_tem, y_tem, length_xy, 50)
        segments_points_xz = arc_segment(x_tem, z_tem, length_xz, 50)

        # the fine parts start from the first spline control points + 49 segment points
        segment_points[0, 0] = x[j]
        segment_points[0, 1] = y[j]
        segment_points[0, 2] = z[j]
        for i in range(segment_number-1):
            segment_points[i+1, 0] = segments_points_xy[i, 0]
            segment_points[i+1, 1] = segments_points_xy[i, 1]
            segment_points[i+1, 2] = segments_points_xz[i, 1]
        spline_fine_point.append(segment_points)
    spline_fine_point = combine_pc(spline_fine_point)
    last_point = np.array([x[-1], y[-1], z[-1]])
    spline_fine_point = np.vstack((spline_fine_point, last_point))
    return spline_fine_point

# Tooth features, explain from which space and in which space
class ToothFeature:
    def __init__(self, narray_n_by_3, tooth_id, from_space='CT', in_space='IOS', voxel_down_sample_size=2):
        self.points = narray_n_by_3
        self.points_amount = narray_n_by_3.shape[0]
        self.id = int(tooth_id)
        self.from_space = from_space
        self.in_space = in_space
        self.o3d_point_cloud, self.centroid, self.eigen_values, self.principal_axes = self.solve_centroid()
        self.ds_point_cloud, self.ds_centroid, self.ds_eigen_values, self.ds_principal_axes = self.down_sampling(voxel_down_sample_size)
        self.ICP = []
        self.local_ICP_transformation = []

    def solve_centroid(self):
        point_cloud = o3d.PointCloud()
        point_cloud.points = o3d.Vector3dVector(self.points)
        centroid, matrix = o3d.compute_point_cloud_mean_and_covariance(point_cloud)
        eigen, axes =np.linalg.eig(matrix)
        return point_cloud, centroid, eigen, axes

    def down_sampling(self, voxel_size):
        pc_ds = o3d.geometry.voxel_down_sample(self.o3d_point_cloud, voxel_size)
        ds_centroid, ds_matrix = o3d.compute_point_cloud_mean_and_covariance(pc_ds)
        ds_eigen, ds_axes = np.linalg.eig(ds_matrix)

        return pc_ds, ds_centroid, ds_eigen, ds_axes


# FullArch class which includes existing and missing (if any) teeth features
class FullArch:
    def __init__(self, missing_tooth_array, from_space='CT', in_space='CT', arch_type='mandible'):
        self.type = arch_type
        self.from_space = from_space
        self.in_space = in_space
        if self.type == 'mandible':
            self.tooth_list = np.asarray(range(17,33,1))
        elif self.type == 'maxillar':
            self.tooth_list = np.asarray(range(1,17,1))
        else:
            raise ValueError('wrong arch type is provided')
        self.missing_tooth_list = np.asarray(missing_tooth_array)
        self.existing_tooth_list = np.setdiff1d(self.tooth_list, self.missing_tooth_list)
        self.existing_tooth = []
        self.missing_tooth = []
        self.spline_points = []     # Centroids of each existing tooth
        self.spline_points_fine = []
        self.missing_spline_points = []     # Centroid of each missing tooth
        self.spline_points_cylindrical = []         # convert spline points to 2D cylindrical coordinate
        self.spline_points_cylindrical_center = []      # center defined by first (17) and last (32) teeth.
        for i in range(len(self.existing_tooth_list)):
            self.existing_tooth.append([])
        for i in range(len(self.missing_tooth_list)):
            self.missing_tooth.append([])
        self.allpoints = []

    def add_tooth(self, tooth_number, tooth_feature):   # add tooth features to corresponding idx.
        if tooth_number in self.existing_tooth_list:
            idx = np.where(self.existing_tooth_list == tooth_number)[0][0]
            self.existing_tooth[idx] = tooth_feature
        elif tooth_number in self.missing_tooth_list:
            idx = np.where(self.missing_tooth_list == tooth_number)[0][0]
            self.missing_tooth[idx] = tooth_feature
        else:
            print('tooth number is out of range')

    def get_tooth(self, tooth_number): # access to the tooth features
        if tooth_number in self.existing_tooth_list:
            idx = np.where(self.existing_tooth_list == tooth_number)[0][0]
            feature = self.existing_tooth[idx]
        elif tooth_number in self.missing_tooth_list:
            idx = np.where(self.missing_tooth_list == tooth_number)[0][0]
            feature = self.missing_tooth[idx]
        else:
            print('tooth '+np.str(tooth_number) + ' is out of range, feature is empty')
            feature = []
        return feature

    def update_spline(self, segment_number=50, fine_flag = False):
        spline_points_tem = []
        for tooth in self.existing_tooth:
            spline_points_tem.append(tooth.centroid)
        self.spline_points = np.asarray(spline_points_tem)
        cylindrical_center = (spline_points_tem[0] + spline_points_tem[-1])/2
        self.spline_points_cylindrical = convert_cylindrical(spline_points_tem, cylindrical_center)
        self.spline_points_cylindrical_center = cylindrical_center
        if fine_flag is True:
            self.spline_points_fine = fit_spline_and_split(self.spline_points, segment_number)
        del spline_points_tem, cylindrical_center

    def get_spline_points(self, tooth_number):
        if tooth_number in self.existing_tooth_list:
            idx = np.where(self.existing_tooth_list == tooth_number)[0][0]
            selected_centroid = self.existing_tooth[idx].centroid
        elif tooth_number in self.missing_tooth_list:
            idx = np.where(self.missing_tooth_list == tooth_number)[0][0]
            selected_centroid = self.missing_tooth[idx].centroid
        return selected_centroid

    def update_all_teeth_points(self):
        all_points = []
        for tooth in self.existing_tooth:
            all_points.append(tooth.points)
        self.allpoints = combine_pc(all_points)
        del all_points

    def update_missing_spline(self):
        missing_spline_points_tem = []
        for tooth in self.missing_tooth:
            missing_spline_points_tem.append(tooth.centroid)
        self.missing_spline_points = np.asarray(missing_spline_points_tem)
        del missing_spline_points_tem

# ------- Perform Feature Selection ------- #

