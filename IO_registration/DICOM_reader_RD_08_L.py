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
    return idx


def trim_slice(dicom_scan, x, y, z):
    trim = []
    surface_idx = []
    for i in range(z[0], z[1]):
        s = dicom_scan[i]
        HU = s.pixel_array[x[0]: x[1], y[0]:y[1]] * slice_properties[-2] + slice_properties[-1]
        HU = HU * slice_properties[-2] + slice_properties[-1]
        slice_bone_surface_idx = extract_bone_surface(HU, THRESHOLD, THRESHOLD + THO_DELTA)
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
path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\RD-08-L\\'
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

# left molar boundary (RD_08_L)
x_molar = [540, 600]
y_molar = [264, 317]
z_molar = [160,175]


# left premolar boundary (RD_08_L)
x_premolar = [540, 600]
y_premolar = [264, 317]
z_premolar = [160,175]

# typodont dicom thresholds
THRESHOLD = 10
THO_DELTA = 50

# dicom slice plot
plot_2D_slice(scan[z_premolar[0] + 5])
plt.show()
exit()

# trim molar and plot
HU_trim_molar, bone_surface_idx_molar = trim_slice(scan, x_molar, y_molar, z_molar)
plot_3D_point_cloud(bone_surface_idx_molar)
del HU_trim_molar

# trim premolar and plot
HU_trim_premolar, bone_surface_idx_premolar = trim_slice(scan, x_premolar, y_premolar, z_premolar)
plot_3D_point_cloud(bone_surface_idx_premolar)

# dicom slice plot
plot_2D_slice(scan[z_molar[0]])
plt.show()
