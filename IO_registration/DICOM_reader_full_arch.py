# Test the DICOM read functions.
# The resference can be found:
# https://hengloose.medium.com/a-comprehensive-starter-guide-to-visualizing-and-analyzing-dicom-images-in-python-7a8430fcb7ed

# common packages
import numpy as np
import os
import copy
from math import *
import matplotlib.pyplot as plt
import Writers as Yomiwrite
from mpl_toolkits.mplot3d import axes3d
from functools import reduce

# reading in dicom files
import pydicom

# import open3d
from open3d import *
import open3d

import cv2


def load_scan_backup(path):
    slices = [pydicom.dcmread(path + '/' + s) for s in
              os.listdir(path)]
    print(np.str(len(slices)) + ' slices were loaded')
    thickness = slices[0].SliceThickness
    pixel_spacing = slices[0].PixelSpacing
    print('slice thickness is ', thickness)
    print('slice pixel spacing is ', pixel_spacing)
    return slices


def load_scan(path):
    scan = []
    n = 0
    #slope = 1
    #intercept = -1000  # Set default value
    for i in range(len(os.listdir(path))):
        #print('i is', i+1)
        #slice = pydicom.dcmread(path + '/' + s)
        slice = pydicom.dcmread(path + "export%d.dcm" %(i+1))
        n += 1
        if n == 1:
            slice_thickness = slice.SliceThickness
            slice_pixel_spacing = slice.PixelSpacing
            slice_rows = slice.Rows
            slice_columns = slice.Columns
            slope = slice.RescaleSlope
            intercept = slice.RescaleIntercept
        #HU_slice = slope * slice.pixel_array + intercept
        #HU.append(HU_slice)
        scan.append(slice)
        del slice
    print('The Scan has ' + np.str(n) + ' slices.')
    print('Each slice has ' + np.str(slice_rows) + ' rows and ' + np.str(slice_columns) + ' columns')
    print('Thickness is ' + np.str(slice_thickness) + ' and pixel spacing is ' + np.str(slice_pixel_spacing) + ' mm')
    return scan, [slice_thickness, slice_pixel_spacing, n, slice_rows, slice_columns, slope, intercept]

def load_scan_human(path):
    HU = []
    scan = []
    n = 0
    slope = 1
    intercept = -1000  # Set default value
    for s in os.listdir(path):
        #print('i is', i+1)
        slice = pydicom.dcmread(path + '/' + s)
        #slice = pydicom.dcmread(path + "export%d.dcm" %(i+1))
        n += 1
        if n == 1:
            slice_thickness = slice.SliceThickness
            slice_pixel_spacing = slice.PixelSpacing
            slice_rows = slice.Rows
            slice_columns = slice.Columns
            slope = slice.RescaleSlope
            intercept = slice.RescaleIntercept
        scan.append(slice)
        del slice
    print('The Scan has ' + np.str(n) + ' slices.')
    print('Each slice has ' + np.str(slice_rows) + ' rows and ' + np.str(slice_columns) + ' columns')
    print('Thickness is ' + np.str(slice_thickness) + ' and pixel spacing is ' + np.str(slice_pixel_spacing) + ' mm')
    return scan, [slice_thickness, slice_pixel_spacing, n, slice_rows, slice_columns, slope, intercept]


# Convert CT pixels to HU units
# reference:
# https://stackoverflow.com/questions/22991009/how-to-get-hounsfield-units-in-dicom-file-using-fellow-oak-dicom-library-in-c-sh
# https://www.medicalconnections.co.uk/kb/Hounsfield-Units/
# https://radiopaedia.org/articles/hounsfield-unit?lang=us
#  Usually, air: -1000 HU;  very dense bone: +-2000; metals: over 3000;
#  High atomic number structures (bone) are white and have a high attenuation value (250–1000 HU
#  air has a low attenuation value (−600 to −1000 HU)
#  fat (−100 HU)
#  muscle has an attenuation value of approximately 50 HU
#  The Hounsfield scale of CT is set around water measuring 0 HU.


# Extract bone surface based on HU values
def extract_bone_surface(hu_array, minimum_hu = 1950, maximum_hu = 2050):
    HU_max = hu_array < maximum_hu
    HU_min = hu_array > minimum_hu
    bone_surface = HU_max & HU_min
    idx = np.where(bone_surface == True)
    idx = np.asarray(idx)
    return idx


def trim_slice(dicom_scan, x, y, z):
    trim = []
    surface_idx = []
    #print('y is', y)

    # flip y along vertical axis
    y_0_new = slice_properties[4] - y[1]
    y_1_new = slice_properties[4] - y[0]
    y[0] = y_0_new
    y[1] = y_1_new
    #print('y is', y)
    for i in range(z[0], z[1]):
        s = dicom_scan[i]
        pixel = cv2.flip(s.pixel_array, 1)
        HU = pixel[x[0]: x[1], y[0]:y[1]] * slice_properties[-2] + slice_properties[-1]
        slice_bone_surface_idx = extract_bone_surface(HU, THRESHOLD, THRESHOLD + THO_DELTA)
        if len(slice_bone_surface_idx[0]) > 0:
            slice_bone_surface_idx[0] = slice_bone_surface_idx[0] + x[0]
            slice_bone_surface_idx[1] = slice_bone_surface_idx[1] + y[0]
        trim.append(HU)
        surface_idx.append(slice_bone_surface_idx)
        del HU, slice_bone_surface_idx
    return trim, surface_idx


def plot_3D_point_cloud(surface_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(surface_idx)):
        if surface_idx[i][0].size > 0:
            x_data = surface_idx[i][1] * slice_properties[1][0]
            # flip y for plotting by (total number of column - current column)
            y_data = (slice_properties[4] - surface_idx[i][0]) * slice_properties[1][0]
            z = i * np.ones(surface_idx[i][1].shape)
            # flip z to re-position head in 3d plot. (CT scans from top to bottom. 3D plotting goes from bot to top)
            z_data = (slice_properties[2] - z) * slice_properties[0]
            ax.scatter(x_data, y_data, z_data, c='r', s=4)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # set the same scale
            # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array(
                [x_data.max() - x_data.min(), y_data.max() - y_data.min(), z_data.max() - z_data.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x_data.max() + x_data.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y_data.max() + y_data.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z_data.max() + z_data.min())
            # Comment or uncomment following both lines to test the fake bounding box:
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')


def open3d_plot(surface_idx):
    points = np.zeros(3)
    for i in range(len(surface_idx)):
        if surface_idx[i][0].size > 0:
            #print('surface_idx is', surface_idx[i])
            x_data = surface_idx[i][1] * slice_properties[1][0]
            # flip y
            y_data = (surface_idx[i][0]) * slice_properties[1][0]
            z = i * np.ones(surface_idx[i][1].shape)
            # flip z to re-position head in 3d plot. (CT scans from top to bottom. 3D plotting goes from bot to top)
            z_data = (slice_properties[2] - z) * slice_properties[0]
            for m in range(x_data.shape[0]):
                pcd_tem = np.array([x_data[m], y_data[m], z_data[m]])
                points = np.vstack((points, pcd_tem))
                del pcd_tem
    points = points[1:,:]
    print('shape of points is', np.shape(points))

    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(points)
    visualization.draw_geometries([pcd])
    return points


def plot_2D_slice(dicom_scan):
    fig = plt.figure()
    HU = dicom_scan.pixel_array * slice_properties[-2] + slice_properties[-1]
    HU_max = HU < THRESHOLD + THO_DELTA
    HU_min = HU > THRESHOLD
    Bone = HU_max & HU_min
    Bone_idx = np.where(Bone == True)
    plt.imshow(HU, cmap=plt.cm.bone)
    plt.scatter(Bone_idx[1], Bone_idx[0], color='r', s=4)  # column and row are switched in plt.scatter and plt.imshow


# Read typodont dicom
#path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\RD-08-L\\'
path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\DICOMs_Typodont\\full_arch_voxel_02\\'
scan, slice_properties = load_scan_human(path)
print('slice_properties are')
print('[slice_thickness, slice_pixel_spacing, n, slice_rows, slice_columns, slice_columns, slope, intercept]')
print(slice_properties)
# [slice_thickness, slice_pixel_spacing, n, slice_rows, slice_columns, slice_columns, slope, intercept]

# Initialize boundary variables for trimming
x_boundary = [0, slice_properties[3]]
y_boundary = [0, slice_properties[4]]
z_boundary = [0, slice_properties[2]]

# Tune boundaries (typodont scans)
#x_boundary = [540, 600]     # y in imshow figure
#y_boundary = [264, 317]     # x in imshow figure
z_boundary = [160,175]

# right molar1 boundary (full arch)
y_Rmolar1 = [200, 243]
x_Rmolar1 = [510, 557]
z_Rmolar1 = [130,175]

# right molar2 boundary (full arch)
y_Rmolar2 = [246, 289]
x_Rmolar2 = [501, 562]
z_Rmolar2 = [130,172]

# right molar3 boundary (full arch)
y_Rmolar3 = [296, 345]
x_Rmolar3 = [494, 557]
z_Rmolar3 = [130,169]

# left molar1 boundary (full arch)
x_Lmolar1 = [240, 320]
y_Lmolar1 = [190, 239]
z_Lmolar1 = [125, 190]

# left molar2 boundary (full arch)
x_Lmolar2 = [255, 311]
y_Lmolar2 = [241, 286]
z_Lmolar2 = [130, 180]

# left molar3 boundary (full arch)
x_Lmolar3 = [258, 320]
y_Lmolar3 = [292, 337]
z_Lmolar3 = [130, 176]

# full boundary (full arch)
x_full = [360, 453]
y_full = [430, 483]
z_full = [130, 176]

# incisor boundary (full arch)
x_incisor = [368, 408]
y_incisor = [434, 478]
z_incisor = [130, 176]


# typodont dicom thresholds
THRESHOLD = -600     # original -350 - 5    # working case -600 200   # -50 60
THO_DELTA = 200

# dicom slice plot
#plot_2D_slice(scan[z_premolar[0] + 10])

# trim incisor and plot
HU_trim_incisor, bone_surface_idx_incisor = trim_slice(scan, x_incisor, y_incisor, z_incisor)
dicom_point_cloud_incisor = open3d_plot(bone_surface_idx_incisor)
dicom_pc_incisor_file = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_incisor.csv'
#dicom_point_cloud_3teeth = np.vstack((dicom_point_cloud_2teeth, dicom_point_cloud_premolar_2))
Yomiwrite.write_csv_matrix(dicom_pc_incisor_file, dicom_point_cloud_incisor)
del HU_trim_incisor



# trim molar and plot
HU_trim_molar, bone_surface_idx_molar = trim_slice(scan, x_Rmolar1, y_Rmolar1, z_Rmolar1)
print('bone_surface_idx_molar is', bone_surface_idx_molar)
dicom_point_cloud = open3d_plot(bone_surface_idx_molar)
dicom_pc_file = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Rmolar1.csv'
#Yomiwrite.write_csv_matrix(dicom_pc_file, dicom_point_cloud)

del HU_trim_molar

# trim premolar and plot
HU_trim_premolar, bone_surface_idx_premolar = trim_slice(scan, x_Rmolar2, y_Rmolar2, z_Rmolar2)
dicom_point_cloud_premolar = open3d_plot(bone_surface_idx_premolar)
dicom_pc_2teeth_file = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Rmolar2.csv'
#dicom_point_cloud_Rmolar2 = np.vstack((dicom_point_cloud, dicom_point_cloud_premolar))
#Yomiwrite.write_csv_matrix(dicom_pc_2teeth_file, dicom_point_cloud_premolar)
del HU_trim_premolar


# trim 2nd premolar and plot
HU_trim_premolar_2, bone_surface_idx_premolar_2 = trim_slice(scan, x_Rmolar3, y_Rmolar3, z_Rmolar3)
dicom_point_cloud_premolar_2 = open3d_plot(bone_surface_idx_premolar_2)
dicom_pc_3teeth_file = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Rmolar3.csv'
#dicom_point_cloud_3teeth = np.vstack((dicom_point_cloud_2teeth, dicom_point_cloud_premolar_2))
#Yomiwrite.write_csv_matrix(dicom_pc_3teeth_file, dicom_point_cloud_premolar_2)
del HU_trim_premolar_2


# trim left molar1 and plot
HU_trim_Lmolar1, bone_surface_idx_Lmolar1 = trim_slice(scan, x_Lmolar1, y_Lmolar1, z_Lmolar1)
dicom_point_cloud_Lmolar1 = open3d_plot(bone_surface_idx_Lmolar1)
dicom_pc_Lmolar1_file = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Lmolar1.csv'
#dicom_point_cloud_3teeth = np.vstack((dicom_point_cloud_2teeth, dicom_point_cloud_premolar_2))
#Yomiwrite.write_csv_matrix(dicom_pc_Lmolar1_file, dicom_point_cloud_Lmolar1)
del HU_trim_Lmolar1

# trim left molar2 and plot
HU_trim_Lmolar2, bone_surface_idx_Lmolar2 = trim_slice(scan, x_Lmolar2, y_Lmolar2, z_Lmolar2)
dicom_point_cloud_Lmolar2 = open3d_plot(bone_surface_idx_Lmolar2)
dicom_pc_Lmolar2_file = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Lmolar2.csv'
#dicom_point_cloud_3teeth = np.vstack((dicom_point_cloud_2teeth, dicom_point_cloud_premolar_2))
#Yomiwrite.write_csv_matrix(dicom_pc_Lmolar2_file, dicom_point_cloud_Lmolar2)
del HU_trim_Lmolar2

# trim left molar3 and plot
HU_trim_Lmolar3, bone_surface_idx_Lmolar3 = trim_slice(scan, x_Lmolar3, y_Lmolar3, z_Lmolar3)
dicom_point_cloud_Lmolar3 = open3d_plot(bone_surface_idx_Lmolar3)
dicom_pc_Lmolar3_file = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_Lmolar3.csv'
#dicom_point_cloud_3teeth = np.vstack((dicom_point_cloud_2teeth, dicom_point_cloud_premolar_2))
#Yomiwrite.write_csv_matrix(dicom_pc_Lmolar3_file, dicom_point_cloud_Lmolar3)
del HU_trim_Lmolar3

# trim full and plot
HU_trim_full, bone_surface_idx_full = trim_slice(scan, x_full, y_full, z_full)
dicom_point_cloud_full = open3d_plot(bone_surface_idx_full)
dicom_pc_full_file = 'G:\My Drive\Project\IntraOral Scanner Registration\dicom_points_full.csv'
#dicom_point_cloud_3teeth = np.vstack((dicom_point_cloud_2teeth, dicom_point_cloud_premolar_2))
#Yomiwrite.write_csv_matrix(dicom_pc_full_file, dicom_point_cloud_full)
del HU_trim_full




# dicom slice plot
# plt.show()
