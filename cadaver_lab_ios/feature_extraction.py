# Prepare features for non-rigid registration
# By Xiao Huang
# 04/05/2021

import open3d as o3d
import numpy as np
import Readers as Yomiread
from scipy import interpolate
import coordinates
import matplotlib.pyplot as plt


def combine_pc(list):
    combined_pc = list[0]
    for i in range(len(list)-1):
        #combined_pc = np.vstack((combined_pc, list[i+1]))
        combined_pc = np.concatenate((combined_pc,list[i+1]))
    return combined_pc


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


# use mid points of two adjacent segment points to guide the translation
def check_boundary(list):
    mid_point = []
    for i in range(len(list)-1):
        mid_point.append((list[i] + list[i+1])/2)
    return np.asarray(mid_point)


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


# Not for general application. Used to generate guided points for only missing teeth
def fit_spline_and_split_fxt(spline_points,  guiding_spline_points, segment_number,):
    x_guiding = guiding_spline_points[:,0]
    y_guiding = guiding_spline_points[:,1]
    z_guiding = guiding_spline_points[:,2]

    spline_points_array = np.asarray(spline_points)
    x = spline_points_array[:,0]
    y = spline_points_array[:,1]
    z = spline_points_array[:,2]

    # Possible errors: https://stackoverflow.com/questions/46816099/scipy-interpolate-splrep-data-error
    #plt.scatter(x, y)
    #plt.show()
    tck_xy = interpolate.splrep(x, y, s=0)
    tck_xz = interpolate.splrep(x, z, s=0)
    deltax = 1e-5
    spline_fine_point = []

    for j in range(len(x_guiding)-1):
        print('refine spline for part', j)
        start = x_guiding[j]
        end = x_guiding[j+1]
        segment_points = np.zeros((segment_number,3))
        x_tem = np.arange(start, end, deltax)
        y_tem = interpolate.splev(x_tem, tck_xy, der=0)
        z_tem = interpolate.splev(x_tem, tck_xz, der=0)
        length_xy = arc_length(x_tem, y_tem)
        length_xz = arc_length(x_tem, z_tem)

        segments_points_xy = arc_segment(x_tem, y_tem, length_xy, 50)
        segments_points_xz = arc_segment(x_tem, z_tem, length_xz, 50)

        # the fine parts start from the first spline control points + 49 segment points
        segment_points[0, 0] = x_guiding[j]
        segment_points[0, 1] = y_guiding[j]
        segment_points[0, 2] = z_guiding[j]
        for i in range(segment_number-1):
            segment_points[i+1, 0] = segments_points_xy[i, 0]
            segment_points[i+1, 1] = segments_points_xy[i, 1]
            segment_points[i+1, 2] = segments_points_xz[i, 1]
        spline_fine_point.append(segment_points)
    spline_fine_point = combine_pc(spline_fine_point)
    last_point = np.array([x_guiding[-1], y_guiding[-1], z_guiding[-1]])
    spline_fine_point = np.vstack((spline_fine_point, last_point))
    return spline_fine_point


# For splint application
def fit_spline_and_split_splint(spline_points, segment_number, base_axis):
    x_g = spline_points[:,0]
    y_g = spline_points[:,1]
    z_g = spline_points[:,2]

    if base_axis == 'y':
        x = y_g
        y = x_g
        z = z_g
    else:
        x = x_g
        y = y_g
        z = z_g

    print('x is', x)
    print('y is', y)

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
    spline_fine_point_final = spline_fine_point
    if base_axis == 'y':
        spline_fine_point_final[:,0] = spline_fine_point[:,1]
        spline_fine_point_final[:,1] = spline_fine_point[:,0]
    print('shape of spline fine is', np.shape(spline_fine_point_final))
    return spline_fine_point_final



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
        self.spline_points_fine = []    #
        self.spline_points_fine_cylindrical = []
        self.spline_points_fine_cylindrical_mid_points = []     # mid points used to guide translation
        self.missing_spline_points = []     # Centroid of each missing tooth
        self.spline_points_cylindrical = []         # convert spline points to 2D cylindrical coordinate
        self.spline_points_cylindrical_center = []      # center defined by first (17) and last (32) teeth.
        self.spline_virtual_points = []
        self.spline_virtual_guided_points = []
        self.spline_virtual_guided_points_fine = []
        self.spline_virtual_guided_points_fine_cylindrical = []
        self.spline_virtual_guided_points_fine_cylindrical_mid_points = []
        self.spline_guided_points = []
        self.spline_guided_points_fine = []
        self.spline_guided_points_fine_cylindrical = []
        self.spline_guided_points_fine_cylindrical_mid_points = []
        for i in range(len(self.existing_tooth_list)):
            self.existing_tooth.append([])
        for i in range(len(self.missing_tooth_list)):
            self.missing_tooth.append([])
            self.spline_virtual_points.append([])
        for i in range(len(self.tooth_list)):
            self.spline_virtual_guided_points.append([])
            self.spline_guided_points.append([])
        self.allpoints = []
        self.ignore_boundary = [-np.pi/2, -np.pi/3]   # initial values are to make sure all points are used by default
        self.target_points = []

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
        self.spline_points_cylindrical = coordinates.convert_cylindrical(spline_points_tem, cylindrical_center)
        self.spline_points_cylindrical_center = cylindrical_center
        if fine_flag is True:
            self.spline_points_fine = fit_spline_and_split(self.spline_points, segment_number)
            self.spline_points_fine_cylindrical = coordinates.convert_cylindrical(self.spline_points_fine, self.spline_points_cylindrical_center)
            self.spline_points_fine_cylindrical_mid_points = check_boundary(self.spline_points_fine_cylindrical[:,1])
        del spline_points_tem, cylindrical_center

    def get_spline_points(self, tooth_number):
        if tooth_number in self.existing_tooth_list:
            idx = np.where(self.existing_tooth_list == tooth_number)[0][0]
            selected_centroid = self.existing_tooth[idx].centroid
        elif tooth_number in self.missing_tooth_list:
            idx = np.where(self.missing_tooth_list == tooth_number)[0][0]
            selected_centroid = self.missing_tooth[idx].centroid
        return selected_centroid

    # if missing_tooth_flag set at True, the missing teeth will be included
    def update_all_teeth_points(self, missing_tooth_flag=False):
        all_points = []
        for tooth in self.existing_tooth:
            all_points.append(tooth.points)
        if missing_tooth_flag == True:
            for tooth in self.missing_tooth:
                all_points.append(tooth.points)
        self.allpoints = combine_pc(all_points)
        del all_points

    def update_missing_spline(self):
        missing_spline_points_tem = []
        for tooth in self.missing_tooth:
            missing_spline_points_tem.append(tooth.centroid)
        self.missing_spline_points = np.asarray(missing_spline_points_tem)
        del missing_spline_points_tem

    def update_ignore_boundary(self, ignore_tooth_list):
        min_theta = []
        max_theta = []
        for i in ignore_tooth_list:
            points_tem = self.get_tooth(i).points
            points_tem_cylindrical = coordinates.convert_cylindrical(points_tem, self.spline_points_cylindrical_center)
            min_theta.append(np.min(points_tem_cylindrical[:,1]))
            max_theta.append(np.max(points_tem_cylindrical[:,1]))
        self.ignore_boundary[0] = np.min(np.asarray(min_theta))
        self.ignore_boundary[1] = np.max(np.asarray(max_theta))

    # add virtual spline points, the virtual spline points are for missing teeth.
    def add_virtual_spline_point(self, tooth_id, virtual_point):
        if tooth_id in self.missing_tooth_list:
            idx = np.where(self.missing_tooth_list == tooth_id)[0][0]
            self.spline_virtual_points[idx] = virtual_point
        else:
            print('tooth number is out of scope')

    def update_virtual_guided_spline(self, segment_number=50, fine_flag = False):
        for i in self.tooth_list:
            if i in self.existing_tooth_list:
                idx = np.where(self.existing_tooth_list == i)[0][0]
                idx2 = np.where(self.tooth_list == i)[0][0]
                self.spline_virtual_guided_points[idx2] = self.spline_points[idx]
            elif i in self.missing_tooth_list:
                idx = np.where(self.missing_tooth_list == i)[0][0]
                idx2 = np.where(self.tooth_list == i)[0][0]
                self.spline_virtual_guided_points[idx2] = self.spline_virtual_points[idx]

        if fine_flag is True:
            points_tem = np.asarray(self.spline_virtual_guided_points)
            fig=plt.figure()
            plt.scatter(points_tem[:,0],points_tem[:,1])
            plt.show()
            self.spline_virtual_guided_points_fine = fit_spline_and_split(points_tem, segment_number)
            self.spline_virtual_guided_points_fine_cylindrical = coordinates.convert_cylindrical(self.spline_virtual_guided_points_fine, self.spline_points_cylindrical_center)
            self.spline_virtual_guided_points_fine_cylindrical_mid_points = check_boundary(self.spline_virtual_guided_points_fine_cylindrical[:,1])

    def update_guided_spline(self, segment_number=50, fine_flag = False):
        for i in self.tooth_list:
            if i in self.existing_tooth_list:
                idx = np.where(self.existing_tooth_list == i)[0][0]
                idx2 = np.where(self.tooth_list == i)[0][0]
                #print('i is', i)
                #print('idx is',idx)
                #print('idx2 is', idx2)
                #print('tooth list is', self.existing_tooth_list)
                #print('number of spline points is', np.shape(self.spline_points))
                #print('spline points are', self.spline_points)
                #print('shape of missing spline points is', np.shape(self.missing_spline_points))
                #print('missing spline points are', self.missing_spline_points)
                #print('soline_points[idx] is', self.spline_points[idx])
                self.spline_guided_points[idx2] = self.spline_points[idx]
            elif i in self.missing_tooth_list:
                idx = np.where(self.missing_tooth_list == i)[0][0]
                idx2 = np.where(self.tooth_list == i)[0][0]
                self.spline_guided_points[idx2] = self.missing_spline_points[idx]

        if fine_flag is True:
            points_tem = np.asarray(self.spline_guided_points)
            self.spline_guided_points_fine = fit_spline_and_split(points_tem, segment_number)
            self.spline_guided_points_fine_cylindrical = coordinates.convert_cylindrical(self.spline_guided_points_fine, self.spline_points_cylindrical_center)
            self.spline_guided_points_fine_cylindrical_mid_points = check_boundary(self.spline_guided_points_fine_cylindrical[:,1])
            del points_tem

    def add_target(self, pc):
        self.target_points = np.asarray(pc)


# fixture class which includes the features of splint fixture.
# For testing purpose, use missing teeth
# Fixture update_spine() will fit spline for all teeth along the arch
class Fixture:
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
        self.spline_points_fine = []    #
        self.spline_points_fine_cylindrical = []
        self.spline_points_fine_cylindrical_mid_points = []     # mid points used to guide translation
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
        for i in self.tooth_list:
            if i in self.existing_tooth_list:
                idx = np.where(self.existing_tooth_list == i)[0][0]
                points_tem = self.existing_tooth[idx].centroid
            elif i in self.missing_tooth_list:
                idx = np.where(self.missing_tooth_list == i)[0][0]
                points_tem = self.missing_tooth[idx].centroid
            spline_points_tem.append(points_tem)
        self.spline_points = np.asarray(spline_points_tem)
        #self.spline_points = spline_points_tem
        cylindrical_center = (spline_points_tem[0] + spline_points_tem[-1])/2
        self.spline_points_cylindrical = coordinates.convert_cylindrical(spline_points_tem, cylindrical_center)
        self.spline_points_cylindrical_center = cylindrical_center

        missing_spline_points_tem = []
        for tooth in self.missing_tooth:
            missing_spline_points_tem.append(tooth.centroid)
        self.missing_spline_points = np.asarray(missing_spline_points_tem)
        del missing_spline_points_tem

        if fine_flag is True:
            self.spline_points_fine = fit_spline_and_split_fxt(self.spline_points, self.missing_spline_points, segment_number)
            self.spline_points_fine_cylindrical = coordinates.convert_cylindrical(self.spline_points_fine, self.spline_points_cylindrical_center)
            self.spline_points_fine_cylindrical_mid_points = check_boundary(self.spline_points_fine_cylindrical[:,1])
        del spline_points_tem, cylindrical_center

    def get_spline_points(self, tooth_number):
        if tooth_number in self.existing_tooth_list:
            idx = np.where(self.existing_tooth_list == tooth_number)[0][0]
            selected_centroid = self.existing_tooth[idx].centroid
        elif tooth_number in self.missing_tooth_list:
            idx = np.where(self.missing_tooth_list == tooth_number)[0][0]
            selected_centroid = self.missing_tooth[idx].centroid
        return selected_centroid

    # if missing_tooth_flag set at True, the missing teeth will be included
    def update_all_teeth_points(self, missing_tooth_flag=False):
        all_points = []
        for tooth in self.existing_tooth:
            all_points.append(tooth.points)
        if missing_tooth_flag == True:
            for tooth in self.missing_tooth:
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


# Splint class which includes the features of the 8 pyramid
# Splint class also includes the 4 selected guiding points
class Splint:
    def __init__(self, spline_guide_points_number, type = 'IOS', spline_base_axis = 'y'):
        self.type = type
        self.spline_fitting_base_axis = spline_base_axis
        self.pyramid_number_list = np.asarray(range(16))
        self.pyramid_fiducial_list = spline_guide_points_number
        self.pyramid_target_list = np.setdiff1d(self.pyramid_number_list, self.pyramid_fiducial_list)

        self.pyramid_points = []
        self.pyramid_fiducial_points = []
        self.pyramid_target_points = []
        self.spline_points_fine = []  #
        self.spline_points_fine_cylindrical = []
        self.spline_points_fine_cylindrical_mid_points = []  # mid points used to guide translation

        self.spline_points_cylindrical = []  # convert spline points to 2D cylindrical coordinate
        self.spline_points_cylindrical_center = []  # center defined by first and last pyramids.

        for i in range(len(self.pyramid_fiducial_list)):
            self.pyramid_fiducial_points.append([])
        for i in range(len(self.pyramid_target_list)):
            self.pyramid_target_points.append([])
        for i in range(len(self.pyramid_number_list)):
            self.pyramid_points.append([])

    def add_pyramid(self, pyramid_number, pyramid_point):  # add tooth features to corresponding idx.
        #print('add fiducial and target pyramids')
        if pyramid_number in self.pyramid_fiducial_list:
            idx = np.where(self.pyramid_fiducial_list == pyramid_number)[0][0]
            self.pyramid_fiducial_points[idx] = pyramid_point
        elif pyramid_number in self.pyramid_target_list:
            idx = np.where(self.pyramid_target_list == pyramid_number)[0][0]
            self.pyramid_target_points[idx] = pyramid_point
        else:
            print('pyramid number is out of range')

        #print('add all pyramids')
        if pyramid_number in self.pyramid_number_list:
            idx = np.where(self.pyramid_number_list == pyramid_number)[0][0]
            self.pyramid_points[idx] = pyramid_point

    def get_pyramid(self, pyramid_number):  # access to pyramid
        if pyramid_number in self.pyramid_fiducial_list:
            idx = np.where(self.pyramid_fiducial_list == pyramid_number)[0][0]
            point = self.pyramid_fiducial_points[idx]
        elif pyramid_number in self.pyramid_target_list:
            idx = np.where(self.pyramid_target_list == pyramid_number)[0][0]
            point = self.pyramid_target_points[idx]
        else:
            print('pyramid ' + np.str(pyramid_number) + ' is out of range, pyramid is empty')
            point = []
        return point

    def update_spline(self, segment_number=50, fine_flag=False):
        spline_points_tem = self.pyramid_fiducial_points
        self.spline_points = np.asarray(spline_points_tem)
        cylindrical_center = (spline_points_tem[0] + spline_points_tem[-1]) / 2
        #cylindrical_center = np.array([5, -16, 233])
        self.spline_points_cylindrical = coordinates.convert_cylindrical(spline_points_tem, cylindrical_center)
        self.spline_points_cylindrical_center = cylindrical_center

        if fine_flag is True:
            self.spline_points_fine = fit_spline_and_split_splint(self.spline_points,
                                                               segment_number, self.spline_fitting_base_axis)
            self.spline_points_fine_cylindrical = coordinates.convert_cylindrical(self.spline_points_fine,
                                                                                  self.spline_points_cylindrical_center)
            self.spline_points_fine_cylindrical_mid_points = check_boundary(self.spline_points_fine_cylindrical[:, 1])
        del spline_points_tem, cylindrical_center

