# Test for new surface extraction code (06/21/2021)

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
import geometry


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
    for i in range(len(os.listdir(path))):
        slice = pydicom.dcmread(path + "export%d.dcm" %(i+1))
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


def load_scan_human(path):
    HU = []
    scan = []
    n = 0
    slope = 1
    intercept = -1000  # Set default value
    for s in os.listdir(path):
        slice = pydicom.dcmread(path + '/' + s)
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
def extract_bone_surface_box(hu_array, box, minimum_hu = 1950, maximum_hu = 2050):
    HU_max = hu_array < maximum_hu
    HU_min = hu_array > minimum_hu
    bone_surface = HU_max & HU_min
    idx = np.where(bone_surface == True)
    delete_idx = []
    for i in range(len(idx[0])):
        # point is defined as [real x , real y] which is [column #, row #]
        # since box is defined as [real x, real y] from matplot_interactive.py
        point_tem = [idx[1][i], idx[0][i]]
        flag = geometry.point_in_polygon(point_tem, box)
        #flag = define_box.quadrangle(point_tem, box)
        if flag == 0:
            delete_idx.append(i)
    new_idx_0 = np.delete(idx[0], delete_idx)   # new row
    new_idx_1 = np.delete(idx[1], delete_idx)   # new column
    new_idx = [new_idx_1, new_idx_0]    # new point is defined as [new column, new row] = [real x, real y]

    del bone_surface, idx, delete_idx
    return new_idx


def trim_slice_box(dicom_scan, slice_properties, box, z_boundary, threshold):
    # box corner is defined as [col_number, row_number]
    trim = []
    surface_idx = []
    #THRESHOLD = 1850
    #THO_DELTA = 400

    #THRESHOLD = -600
    #THO_DELTA = 200

    for i in range(z_boundary[0], z_boundary[1]):
        s = dicom_scan[i]
        pixel = s.pixel_array
        HU = pixel * slice_properties[-2] + slice_properties[-1]
        slice_bone_surface_idx = extract_bone_surface_box(HU, box, threshold[0], threshold[1])
        trim.append(HU)
        surface_idx.append(slice_bone_surface_idx)
        del HU, slice_bone_surface_idx, pixel, s
    return trim, surface_idx


def trim_tooth_box(scan, slice_properties, tooth_number, z_boundary, box, threshold, surface_name):
    i = tooth_number
    z_boundary = z_boundary
    #z_boundary = BOUNDARY[i-1,4:].astype(int)
    box = box
    #box = BOX[i-17]
    HU_trim, bone_surface_idx = trim_slice_box(scan, slice_properties, box, z_boundary, threshold)
    dicom_point_cloud = open3d_plot(bone_surface_idx, slice_properties, z_boundary)
    dicom_pc_file = 'C:\\tooth segmentation raw\\tooth_' + np.str(i+1) + '_surface_' + surface_name + '.csv'
    Yomiwrite.write_csv_matrix(dicom_pc_file, dicom_point_cloud)
    del HU_trim, bone_surface_idx, dicom_point_cloud


def open3d_plot(surface_idx, slice_properties, z_boundary):
    points = np.zeros(3)
    n = 0
    for i in range(len(surface_idx)):
        if surface_idx[i][0].size > 0:
            #print('surface_idx is', surface_idx[i])
            #x_data = (slice_properties[4]- surface_idx[i][1]) * slice_properties[1][0]
            x_data = surface_idx[i][0] * slice_properties[1][1]  # Yomiplan x
            # flip y
            #y_data = (surface_idx[i][0]) * slice_properties[1][0]
            y_data = (slice_properties[4] - surface_idx[i][1]) * slice_properties[1][0]  # Yomiplan y

            z = i * np.ones(surface_idx[i][1].shape)
            # flip z to re-position head in 3d plot. (CT scans from top to bottom. 3D plotting goes from bot to top)
            #z_data = (slice_properties[2] - z) * slice_properties[0]  # for reversed
            z_data = (slice_properties[2] - (z_boundary[0] + z)) * slice_properties[0]
            #z_data = z * slice_properties[0]

            if n == 0:
                print('x_data is', x_data)
                print('y_data is', y_data)
                print('z_data is', z_data)
                print('x idx is', surface_idx[i][1])
                print('y idx is', surface_idx[i][0])
                print('z idx is', z)
                n = 1

            #z_data = (z) * slice_properties[0]
            for m in range(x_data.shape[0]):
                pcd_tem = np.array([x_data[m], y_data[m], z_data[m]])
                points = np.vstack((points, pcd_tem))
                del pcd_tem
        #del x_data, y_data, z_data
    points = points[1:,:]
    #print('points [0] are', points[0,:])

    #pcd = open3d.PointCloud()
    #pcd.points = open3d.Vector3dVector(points)
    #visualization.draw_geometries([pcd])
    return points


# Find the point with the closet intensity value to the medium of upper and lower bounds
def select_dicom_surface_points(intensity_lower_bound, intensity_upper_bound, list_of_intensity):
    intensity_mid = (intensity_lower_bound + intensity_upper_bound) / 2
    intensity_err = [ele-intensity_mid for ele in list_of_intensity]
    intensity_err_abs = [abs(ele) for ele in intensity_err]
    idx = intensity_err_abs.index(min(intensity_err_abs))
    return idx





# # Read typodont dicom
# #path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\RD-08-L\\'
# #path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\DICOMs_Typodont\\full_arch_voxel_02_two_holes\\scan1\\'
# #path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\DICOMs_Typodont\\full_arch_voxel_02\\'
# path = "G:\\My Drive\\Project\\IntraOral Scanner Registration\\DICOMs_Typodont\\drill_fixture\\fxt_trial2"
# scan, slice_properties = load_scan_human(path)
# print('slice_properties are')
# print('[slice_thickness, slice_pixel_spacing, n, slice_rows, slice_columns, slice_columns, slope, intercept]')
# print(slice_properties)
# # [slice_thickness, slice_pixel_spacing, n, slice_rows, slice_columns, slice_columns, slope, intercept]
#
# tooth_number = np.asarray(range(32, 16, -1))
# # typodont dicom thresholds
# THRESHOLD = -600     # original -350 - 5    # working case -600 200   # -50 60
# THO_DELTA = 200
# # boundary in [xmin, xmax, ymin, ymax, zmin, zmax]
# BOUNDARY = np.zeros((32,6))
#
# # z has to start with the same values (Bug1)
# X = 130
# BOUNDARY[32-1,:] = np.array([510, 564, 200, 242, X, 179])
# BOUNDARY[31-1,:] = np.array([510, 564, 247, 290, X, 172])
# BOUNDARY[30-1,:] = np.array([494, 557, 296, 345, X, 169])
# BOUNDARY[29-1,:] = np.array([494, 540, 350, 375, X, 166])
# BOUNDARY[28-1,:] = np.array([480, 528, 383, 411, X, 167])
# BOUNDARY[27-1,:] = np.array([474, 510, 418, 443, X, 166])
# BOUNDARY[26-1,:] = np.array([451, 474, 439, 468, X, 172])
# BOUNDARY[25-1,:] = np.array([409, 445, 436, 482, X, 172])
# BOUNDARY[24-1,:] = np.array([368, 407, 436, 482, X, 176])
# BOUNDARY[23-1,:] = np.array([340, 364, 435, 468, X, 176])
# BOUNDARY[22-1,:] = np.array([311, 338, 411, 441, X, 174])
# BOUNDARY[21-1,:] = np.array([294, 335, 377, 403, X, 174])
# BOUNDARY[20-1,:] = np.array([274, 324, 345, 371, X, 174])
# BOUNDARY[19-1,:] = np.array([258, 319, 292, 340, X, 177])
# BOUNDARY[18-1,:] = np.array([255, 312, 243, 285, X, 185])
# BOUNDARY[17-1,:] = np.array([240, 320, 190, 239, X, 188])
#
# # corner points are defined as [col#, row#]
# #p0_32 = [200, 510]      # typodont_2
# #p1_32 = [237, 511]      # typodont 2
# #p2_32 = [250, 563]      # typodont 2
# #p3_32 = [202, 559]      # typodont 2
# #BOX_32 = [p0_32, p1_32, p2_32, p3_32]       # typodont 2
#
# p0_32 = [200, 510]      # fxt_2
# p1_32 = [237, 511]      # tfxt 2
# p2_32 = [250, 563]      # typodont 2
# p3_32 = [202, 559]      # typodont 2
# BOX_32 = [p0_32, p1_32, p2_32, p3_32]       # typodont 2
#
# # tooth 26
# p0_26 = [425, 461]
# p1_26 = [426, 437]  # 440, 442
# p2_26 = [472, 453]
# p3_26 = [465, 494]
# BOX_26 = [p0_26, p1_26, p2_26, p3_26]
#
# # tooth 27
# p0_27 = [386, 454]
# p1_27 = [425, 461]
# p2_27 = [465, 494]
# p3_27 = [424, 517]
# BOX_27 = [p0_27, p1_27, p2_27, p3_27]
#
# # tooth 28
# p0_28 = [367, 483]
# p1_28 = [386, 454]
# p2_28 = [424, 517]
# p3_28 = [391, 539]
# BOX_28 = [p0_28, p1_28, p2_28, p3_28]
#
# # tooth 29
# p0_29 = [340, 488]
# p1_29 = [367, 483]
# p2_29 = [391, 539]
# p3_29 = [356, 575]
# BOX_29 = [p0_29, p1_29, p2_29, p3_29]
#
# # tooth 30
# p0_30 = [286, 500]
# p1_30 = [340, 488]
# p2_30 = [356, 575]
# p3_30 = [302, 574]
# BOX_30 = [p0_30, p1_30, p2_30, p3_30]
#
# # tooth 31
# p0_31 = [237, 511]
# p1_31 = [286, 500]
# p2_31 = [302, 574]
# p3_31 = [250, 563]
# BOX_31 = [p0_31, p1_31, p2_31, p3_31]
#
# # tooth 25
# p0_25 = [426, 437]
# p1_25 = [433, 407]
# p2_25 = [492, 407]
# p3_25 = [472, 453]
# BOX_25 = [p0_25, p1_25, p2_25, p3_25]
#
# # tooth 24
# p0_24 = [433, 407]
# p1_24 = [426, 377]
# p2_24 = [476, 358]
# p3_24 = [492, 407]
# BOX_24 = [p1_24, p2_24, p3_24, p0_24]
#
# # tooth 23
# p0_23 = [426, 377]
# p1_23 = [418, 356]
# p2_23 = [453, 327]
# p3_23 = [476, 358]
# BOX_23 = [p1_23, p2_23, p3_23, p0_23]
#
# # tooth 22
# p0_22 = [418, 356]
# p1_22 = [392, 340]
# p2_22 = [425, 287]
# p3_22 = [453, 327]
# BOX_22 = [p1_22, p2_22, p3_22, p0_22]
#
# # tooth 21
# p0_21 = [392, 340]
# p1_21 = [363, 336]
# p2_21 = [384, 274]
# p3_21 = [425, 287]
# BOX_21 = [p1_21, p2_21, p3_21, p0_21]
#
# # tooth 20
# p0_20 = [363, 336]
# p1_20 = [333, 325]
# p2_20 = [349, 259]
# p3_20 = [384, 274]
# BOX_20 = [p1_20, p2_20, p3_20, p0_20]
#
# # tooth 19
# p0_19 = [333, 325]
# p1_19 = [277, 321]
# p2_19 = [302, 237]
# p3_19 = [349, 259]
# BOX_19 = [p1_19, p2_19, p3_19, p0_19]
#
# # tooth 18
# p0_18 = [277, 321]
# p1_18 = [234, 310]
# p2_18 = [248, 241]
# p3_18 = [302, 237]
# BOX_18 = [p1_18, p2_18, p3_18, p0_18]
#
# # tooth 17
# p0_17 = [234, 310]
# p1_17 = [186, 298]
# p2_17 = [202, 235]
# p3_17 = [248, 241]
# BOX_17 = [p1_17, p2_17, p3_17, p0_17]
#
# BOX = [BOX_17, BOX_18, BOX_19, BOX_20, BOX_21, BOX_22, BOX_23, BOX_24, BOX_25, BOX_26, BOX_27, BOX_28, BOX_29, BOX_30, BOX_31, BOX_32]

#trim_tooth_box(scan, 17)
#exit()

# for i in tooth_number:
#     print('i is', i)
     #trim_tooth_box(scan,i)
# exit()