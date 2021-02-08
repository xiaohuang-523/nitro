# Test the DICOM read functions.
# The resference can be found:
# https://hengloose.medium.com/a-comprehensive-starter-guide-to-visualizing-and-analyzing-dicom-images-in-python-7a8430fcb7ed

# common packages
import numpy as np
import os
import copy
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from functools import reduce

# reading in dicom files
import pydicom


# skimage image processing packages
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops

# scipy linear algebra functions
from scipy.linalg import norm
import scipy.ndimage

# ipywidgets for some interactive plots
from ipywidgets.widgets import *
import ipywidgets as widgets

# plotly 3D interactive graphs
import plotly
from plotly.graph_objs import *
# import chart_studio.plotly as py


# Analyze


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
    HU = []
    scan = []
    n = 0
    slope = 1
    intercept = -1000  # Set default value
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
        #HU_slice = slope * slice.pixel_array + intercept
        #HU.append(HU_slice)
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
def get_pixels_hu_backup(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def get_pixels_hu(scans):
    slope = scans[0].RescaleSlope
    intercept = scans[0].RescaleIntercept
    HU = []
    for scan in scans:
        scan_hu = scan.pixel_array * slope + intercept
        HU.append(scan_hu)
    return np.asarray(HU)


# Extract bone surface based on HU values
def extract_bone_surface(HU_array, minimum_HU = 1950, maximum_HU = 2050):
    HU_max = HU_array < maximum_HU
    HU_min = HU_array > minimum_HU
    bone_surface = HU_max & HU_min
    bone_surface_idx = np.where(bone_surface == True)
    return bone_surface_idx


def trim_slice(bone_surface_idx, x_boundary, y_boundary):
    x_data = bone_surface_idx[1]
    y_data = bone_surface_idx[0]
    x = []
    y = []
    if len(x_data) > 0:
        for i in range(len(x_data)):
            if x_boundary[0] <= x_data[i] <= x_boundary[1] and y_boundary[0] <= y_data[i] <= y_boundary[1]:
                x.append(x_data[i])
                y.append(y_data[i])
    trim_bone_surface_idx = [x,y]
    return trim_bone_surface_idx


# Read typodont dicom
path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\MG0\\'
scan, slice_properties = load_scan(path)

# Read human dicom
# path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\DICOMs_Human\\Bravo One\\'
# scan, slice_properties = load_scan_human(path)

# [slice_thickness, slice_pixel_spacing, n, slice_rows, slice_columns, slice_columns, slope, intercept]
print('scan is', len(scan))
bone_surface_idx = []
x_boundary = [0, slice_properties[3]]
y_boundary = [0, slice_properties[4]]
z_boundary = [0, slice_properties[2]]
#z_boundary = []

# hunman dicom boundary
#x_boundary = [320, 365]     # y in imshow figure
#y_boundary = [350, 450]     # x in imshow figure
#z_boundary = [140,155]

# typodont boundary
x_boundary = [40, 110]     # y in imshow figure
y_boundary = [slice_properties[4]-320, slice_properties[4]-240]     # x in imshow figure
z_boundary = [280,310]


# Threshold values
# Fiducials:        1100 - 1300
# Typodont tooth:   200-300
# Bone:             2000 - 2500

# human dicom thresholds
#threshold = 1800
#th_delta = 600

# typodont dicom thresholds
threshold = 0
th_delta = 150
print('slope is', slice_properties[-2])
print('intercept is', slice_properties[-1])
#exit()
for s in scan:
    HU = s.pixel_array[x_boundary[0] : x_boundary[1], y_boundary[0]:y_boundary[1]] * slice_properties[-2] + slice_properties[-1]
    #HU = s.pixel_array * slice_properties[-2] + slice_properties[-1]
    slice_bone_surface_idx = extract_bone_surface(HU, threshold, threshold + th_delta)
    #slice_bone_surface_idx = extract_bone_surface(HU)
    bone_surface_idx.append(slice_bone_surface_idx)
    del HU

# fig1 = plt.figure()
# plt.imshow(scan[20].pixel_array, cmap = plt.cm.bone)
# fig2 = plt.figure()
# plt.imshow(scan[40].pixel_array, cmap = plt.cm.bone)
# fig3 = plt.figure()
# plt.imshow(scan[60].pixel_array, cmap = plt.cm.bone)
# fig4 = plt.figure()
# plt.imshow(scan[80].pixel_array, cmap = plt.cm.bone)
# plt.show()

delta_z = z_boundary[1] - z_boundary[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(z_boundary[0], z_boundary[1]):
    if bone_surface_idx[i][0].size > 0:
        x_data = bone_surface_idx[i][1] * slice_properties[1][0]
        y_data = bone_surface_idx[i][0] * slice_properties[1][0]
        z = i * np.ones(bone_surface_idx[i][1].shape)
        z_data = z * slice_properties[0]
        ax.scatter(x_data,y_data,z_data, c='r', s=4)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # set the same scale
        # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([x_data.max() - x_data.min(), y_data.max() - y_data.min(), z_data.max() - z_data.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x_data.max() + x_data.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y_data.max() + y_data.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z_data.max() + z_data.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')



fig = plt.figure()
slice1 = pydicom.dcmread(path + '/export295.dcm') # read typodont
#slice1 =pydicom.dcmread(path + '/0144.dcm') # read human
slope = slice1.RescaleSlope
intercept = slice1.RescaleIntercept
HU = slope * slice1.pixel_array + intercept
HU_max = HU < threshold + th_delta
HU_min = HU > threshold
Bone = HU_max & HU_min
Bone_idx = np.where(Bone == True)
print('bone idx is ', Bone_idx)

#HU2 = slope * scan[359].pixel_array + intercept
#print('H2 size is', np.shape(HU2))
#HU2_max = HU2 < 2050
#HU2_min = HU2 > 1950
#Bone2 = HU2_max & HU2_min
#Bone_idx2 = np.where(Bone2 == True)
#print('bone idx check is ', Bone_idx2)



plt.imshow(HU, cmap=plt.cm.bone)
plt.scatter(Bone_idx[1], Bone_idx[0], color = 'r', s=4) # column and row are switched in plt.scatter and plt.imshow
#plt.imshow(np.asarray(Bone_idx),c='red')
plt.show()
#exit()
#Bone_min = np.where(HU > 1950)
#Bone = np.intersect1d(Bone_max, Bone_min)
#print('Bone_max is', Bone_max)

#plt.imshow(HU, cmap=plt.cm.bone)
#plt.imshow(slice1.pixel_array, cmap=plt.cm.bone)
#plt.show()