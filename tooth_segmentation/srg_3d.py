import numpy as np
import matplotlib.pyplot as plt
import DICOM_reader as DR
import cv2
import plot as Yomiplot
import array_processing as ap
from tqdm import tqdm
import datetime
import coordinates

# Reference:
# 1. python code (https://github.com/SamarthGupta93/Interactive-Region-Growing-Segmentation)
# 2. python and C code (http://notmatthancock.github.io/2017/10/09/region-growing-wrapping-c.html)

# SRG method
def srg_3d(region_candidate, ct_file, z_threshold):
    is_srg_finished = False
    #
    i = 0
    while not is_srg_finished:
        # at step i
        print('SRG at step ', i+1)
        if i > 0:
            # update boundary lists using the information from the previous step
            # update values 1 step behind assures that the data is not updated when the check fails. Namely when no
            #               new boundary points are found, the current boundary is not updated.
            region_candidate.update_boundary(region_candidate.boundary_candidate)
        print('pc size is', np.shape(region_candidate.boundary))
        for point in region_candidate.boundary:  # check the neighbour points of all boundary points
            #if point not in region_candidate.region:
            region_candidate.region.append(point)
            if i < 1:
                region_point_tem = region_point_3D(point)
            else:
                region_point_tem = region_point_3D_vector(point, SEED)

            #for neighbour_point_tem in region_point_tem.neighbour:
            #    if neighbour_point_tem[2] > z_threshold - 1: # if the point is above the z_threshold
            #        if neighbour_point_tem not in region_candidate.boundary_p:  # if it's not in region already
            #            if intensity_3d(ct_file, neighbour_point_tem) > 800:   # if intensity is qualified
            #                region_candidate.boundary_candidate.append(neighbour_point_tem) # add to boundary_candidate
            #                region_candidate.remove_boundary_candidate_duplicate()

            #for neighbour_point_tem in region_point_tem.neighbour:
            #    if neighbour_point_tem[2] > z_threshold - 1: # if the point is above the z_threshold
            #        if neighbour_point_tem not in region_candidate.boundary_p:  # if it's not in region already
            #            intensity_diff = intensity_3d(ct_file, neighbour_point_tem) - intensity_3d(ct_file, point)
            #            if intensity_diff > 0:   # if intensity is increasing, count the point
            #                region_candidate.boundary_candidate.append(neighbour_point_tem)  # add to boundary_candidate
            #                region_candidate.remove_boundary_candidate_duplicate()
            #            elif intensity_diff > CUTOFF:   # if intensity is decreasing, the maximum difference is 200
            #                                            # (-200 < diff < 0)
            #                if intensity_3d(ct_file, neighbour_point_tem) > 600:  # if intensity is qualified
            #                    region_candidate.boundary_candidate.append(neighbour_point_tem) # add to boundary_candidate
            #                    region_candidate.remove_boundary_candidate_duplicate()

            #check all neighbours, if they are not in the region already, check if they are qualified for tooth.
            for neighbour_point_tem in region_point_tem.neighbour:
                if neighbour_point_tem[2] > z_threshold - 1: # if the point is above the z_threshold
                    # if neighbour_point_tem not in region_candidate.region:  # if it's not in region already
                    if neighbour_point_tem not in region_candidate.boundary_p:  # if it's not in region already
                        #intensity_diff = intensity_3d(ct_file, neighbour_point_tem) - intensity_3d(ct_file, point)
                        #if intensity_diff > 0:  # if intensity is increasing, count the point
                        #    if neighbour_point_tem not in region_candidate.boundary_candidate:  # do not add duplicates
                        #        region_candidate.boundary_candidate.append(neighbour_point_tem)  # add to boundary_candidate
                        #        region_candidate.remove_boundary_candidate_duplicate()
                        #if intensity_diff > CUTOFF:
                        if intensity_3d(ct_file, neighbour_point_tem) > 700:   # if intensity is qualified
                            if neighbour_point_tem not in region_candidate.boundary_candidate: # do not add duplicates
                                region_candidate.boundary_candidate.append(neighbour_point_tem) # add to boundary_candidate

        if len(region_candidate.boundary_candidate) < 1:  # if no qualified points found, terminate SRG
            is_srg_finished = True

        if i == SRG_LOOPS:
            is_srg_finished = True

        del region_point_tem
        i += 1
    return True


# function to get intensity values from CT scan
def intensity_3d(scan, point_location):
    x = point_location[0]
    y = point_location[1]
    z = point_location[2]
    img = scan[z].pixel_array[230:600, 200:620]
    HU = img * slice_properties[-2] + slice_properties[-1]
    intensity_value = HU[x, y]
    return intensity_value


# define 3D point class (seed) which has properties self and neighbour
class region_point_3D:
    def __init__(self, point):
        point_array = np.asarray(point)
        #if point_array.ndim == 1 and len(point_array) == 3: # if point is a 3d point vector
        self.point = point
        self.neighbour = []    # generate neighbours
        self.neighbour.append((point_array - np.array([1, 0, 0])).tolist())  # -x
        self.neighbour.append((point_array + np.array([1, 0, 0])).tolist())  # +x
        self.neighbour.append((point_array - np.array([0, 1, 0])).tolist())  # -y
        self.neighbour.append((point_array + np.array([0, 1, 0])).tolist())  # +y
        self.neighbour.append((point_array - np.array([0, 0, 1])).tolist())  # -z
        self.neighbour.append((point_array + np.array([0, 0, 1])).tolist())  # +z
        #else:
        #    raise ValueError('The input point should be a 3d vector')


class region_point_3D_vector:
    def __init__(self, point, origin):
        point_array = np.asarray(point)
        vec_origin = (point_array - np.asarray(origin)).tolist()
        vec_neighbours = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        self.point = point
        self.neighbour = []
        for vec in vec_neighbours:
            angle_tem = coordinates.angle_of_two_vec(vec, vec_origin)
            #print('origin is', origin)
            #print('vec is', vec)
            #print('angle_tem is', angle_tem)
            if angle_tem < (np.pi/180 * 140): # only select point which gives <90 degree angle.
                self.neighbour.append((point_array + np.asarray(vec)).tolist())


# define region class for SRG
class region:
    def __init__(self, point):
        print('create 3D region')
        seed = region_point_3D(point)
        self.region = []
        self.boundary = []              # define the current boundary points (boundary at step i)
        self.boundary_p = []            # define the points added in boundary (boundary at step i-1)
        self.boundary_pp = []           # define the points added in boundary (boundary at step i-2)
        self.boundary_candidate = []    # define the points to be added to boundary (boundary at step i+1)
        self.boundary_final = []
        self.center = seed.point

        # initialize boundary, pre-boundary, region and center with the seed point
        self.boundary.append(seed.point)
        self.boundary_p.append(seed.point)
        self.region.append(seed.point)

    # update final boundary
    def update_final_boundary(self, z_threshold):
        print('updating tooth surface')
        self.remove_region_duplicate()
        for point in self.region:
            count = 0
            region_point_tem = region_point_3D(point)
            for neighbour_tem in region_point_tem.neighbour:
                if neighbour_tem in self.region:
                    count += 1
            if count < 6 and point[2] > z_threshold:
                self.boundary_final.append(point)

    def remove_boundary_candidate_duplicate(self):
        list_tem = ap.remove_duplicate_lists(self.boundary_candidate)
        self.boundary_candidate.clear()
        self.boundary_candidate = list_tem.copy()

    def remove_region_duplicate(self):
        list_tem = ap.remove_duplicate_lists(self.region)
        self.region.clear()
        self.region = list_tem.copy()

    def clear_final_boundary(self):
        self.boundary_final.clear()

    # update boundary
    # Note: 3 layers of boundary are defined for potential registration usage.
    # update the current boundary with the candidate boundary (candidate_point_list)
    # set the current boundary to the previous boundary (boundary_pre)
    # set the boundary_pre to boundary_pre_pre
    #
    def update_boundary(self, candidate_point_list):
        # update boundary_pp
        self.boundary_pp.clear()
        self.boundary_pp = self.boundary_p.copy()

        # update boundary_p
        self.boundary_p.clear()
        self.boundary_p = self.boundary.copy()

        # update boundary
        self.boundary.clear()
        self.boundary = self.boundary_candidate.copy()
        self.boundary_candidate.clear()

        # update region
        #for point in self.boundary:
            #if point not in self.region:
                #self.region.append(point)




path_point_surface_matching = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\CT tooth segmentation\\CASE 5277\\CT3\\"
#path_point_surface_matching = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\HW-9898 Point-to-surface experiments\\CT scan\\drill_fxt_0.2\\"
# path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\FA-human scan\\'
scan, slice_properties = DR.load_scan_human(path_point_surface_matching)
fig = plt.figure()

img = scan[265].pixel_array[230:600, 200:620]
#img = scan[260].pixel_array
HU = img * slice_properties[-2] + slice_properties[-1]
test_img = np.zeros((HU.shape[0], HU.shape[1]))

HU_data = HU.flatten()
img_data = img.flatten()
HU_data_square = np.square(HU_data)
y, bins, p = plt.hist(HU_data, bins=200, density=True, alpha=0.6, color='g')

plt.figure()
plt.imshow(HU)
plt.show()

#SEED = [94, 267, 265]  # tooth 21
#SEED = [59, 246, 265]  # tooth 22
SEED = [316, 354, 265] # tooth 17
#[323, 360, 285]
z_tho = 230
SRG_LOOPS = 45
CUTOFF = -200

region_candidate = region(SEED)
srg_3d(region_candidate, scan, z_threshold=z_tho)
region_candidate.update_final_boundary(z_threshold=z_tho)

import open3d as o3d

source = o3d.PointCloud()
source.points = o3d.Vector3dVector(region_candidate.boundary_final)
source.paint_uniform_color([0, 0.651, 0.929])  # yellow
o3d.draw_geometries([source])


#plt.figure()
#ax = plt.axes(projection='3d')
#print('shape of boundary is', np.shape(region_candidate.boundary_final))
#Yomiplot.plot_3d_points(region_candidate.boundary_final, ax, color='green', alpha=0.3, axe_option=False)

#plt.show()


exit()


#img2 = cv2.imread(HU)
HU_square = np.square(HU)

# Apply bilateral filter with d = 15,
# sigmaColor = sigmaSpace = 75.
HU_f32 = HU.astype(np.float32)

#print('HU_f32 is', HU_f32)
#print('HU is', HU)
#print('HU square is', HU_square)
print('maximum of HU is', np.max(HU_f32))
bilateral = cv2.bilateralFilter(HU_f32, 15, 75, 75)
diff_total = []
for j in range(8):
    diff = []
    for i in range(bilateral.shape[1]-1):
        diff.append(bilateral[35 + j,i+1] - bilateral[35 + j, i])
    diff_total.append(diff)
    del diff

fig3 = plt.figure()
#plt.scatter(range(HU_f32.shape[1]), HU_f32[40, :], label='original HU')
plt.scatter(range(bilateral.shape[1]), bilateral[40, :], label='filterd HU')
plt.scatter(range(bilateral.shape[1]-1), diff_total[6], label='pixel difference')
#for j in range(8):
#    plt.scatter(range(bilateral.shape[1]-1), diff_total[j], label='pixel difference')
#print('bilateral is', bilateral)
fig2 = plt.figure()
plt.imshow(bilateral)
#bilateral2 = cv2.bilateralFilter(HU_f32, 15, 25, 75)
#print('bilateral2 is', bilateral2)
#fig3 = plt.figure()
#plt.imshow(bilateral2)

#fig4 = plt.figure()
#plt.imshow(HU_square)
#fig5 = plt.figure()
#bilateral3 = cv2.bilateralFilter(HU_square.astype(np.float32), 15, 25, 75)
#plt.imshow(bilateral3)


# Save the output.
#cv2.imwrite('taj_bilateral.jpg', bilateral)

plt.legend()
plt.show()

