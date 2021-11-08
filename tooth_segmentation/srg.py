import numpy as np
import matplotlib.pyplot as plt
import DICOM_reader as DR
import cv2
import plot as Yomiplot
import array_processing as ap
import coordinates


# SRG method
def srg(region_candidate, ct_file, slice_idx, dim=2):
    is_srg_finished = False
    #
    i = 0
    while not is_srg_finished:
        # at step i
        print('SRG at step ', i+1)
        time1 = np.datetime64('now')
        if i > 0:
            # update boundary lists using the information from the previous step
            # update values 1 step behind assures that the data is not updated when the check fails. Namely when no
            #               new boundary points are found, the current boundary is not updated.
            region_candidate.update_boundary(region_candidate.boundary_candidate)

        for point in region_candidate.boundary:  # check the neighbour points of all boundary points
            if dim == 2:
                region_point_tem = region_point_2D(point)
                #if i < 1:
                #    region_point_tem = region_point_2D(point)
                #else:
                #    region_point_tem = region_point_2D_vector(point, SEED)
            else:
                region_point_tem = region_point_3D(point)
            #print('point is', point)
            #print('neighbour is', region_point_tem.neighbour)

            # check all neighbours, if they are not in the region already, check if they are qualified for tooth.
            for neighbour_point_tem in region_point_tem.neighbour:
                #if neighbour_point_tem not in region_candidate.region:  # if it's not in region already
                if neighbour_point_tem not in region_candidate.boundary_p:  # if it's not in region already
                    if intensity(ct_file, neighbour_point_tem, slice_idx, dim) > 600:   # if intensity is qualified
                        if neighbour_point_tem not in region_candidate.boundary_candidate: # do not add duplicates
                            region_candidate.boundary_candidate.append(neighbour_point_tem) # add to boundary_candidate

        if len(region_candidate.boundary_candidate) < 1:  # if no qualified points found, terminate SRG
            is_srg_finished = True

        if i == 60:
            is_srg_finished = True
        i += 1
        time2 = np.datetime64('now')
        print('calculation time is', time2 - time1)
    return True


# SRG method
def srg_3d(region_candidate, ct_file, dim=3):
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

        for point in region_candidate.boundary:  # check the neighbour points of all boundary points
            if dim == 2:
                region_point_tem = region_point_2D_vector(point)
            else:
                region_point_tem = region_point_3D(point)
            print('point is', point)
            print('neighbour is', region_point_tem.neighbour)
            # check all neighbours, if they are not in the region already, check if they are qualified for tooth.
            for neighbour_point_tem in region_point_tem.neighbour:
                #if neighbour_point_tem not in region_candidate.region:  # if it's not in region already
                if neighbour_point_tem not in region_candidate.boundary_p:  # if it's not in region already
                    if intensity_3d(ct_file, neighbour_point_tem) > 800:   # if intensity is qualified
                        if neighbour_point_tem not in region_candidate.boundary_candidate: # do not add duplicates
                            region_candidate.boundary_candidate.append(neighbour_point_tem) # add to boundary_candidate

        if len(region_candidate.boundary_candidate) < 1:  # if no qualified points found, terminate SRG
            is_srg_finished = True

        if i == 80:
            is_srg_finished = True
        i += 1
    return True



# function to get intensity values from CT scan
def intensity(scan, point_location, slice_idx, dim):
    if dim == 2:
        img = scan[slice_idx].pixel_array[230:600, 200:620]
        #img = scan[slice_idx].pixel_array
        HU = img * slice_properties[-2] + slice_properties[-1]
        #plt.imshow(HU)
        #plt.show()
        x = point_location[0]
        y = point_location[1]
        #print('x is', x)
        #print('y is', y)
        intensity_value = HU[x, y]
        #print('intensity is', intensity_value)
    #else:
    #    x = point_location[0]
    #    y = point_location[1]
    #    z = point_location[2]
    #    img = scan[z].pixel_array[280:440, 270:500]
    #    HU = img * slice_properties[-2] + slice_properties[-1]
    #    intensity_value = HU[x, y]

    else:
        x = point_location[0]
        y = point_location[1]
        img = scan[slice_idx].pixel_array[230:600, 200:620]
        #img = scan[slice_idx].pixel_array
        HU = img * slice_properties[-2] + slice_properties[-1]
        intensity_value = HU[x, y]

    return intensity_value


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
        if point_array.ndim == 1 and len(point_array) == 3: # if point is a 3d point vector
            self.point = point
            self.neighbour = []    # generate neighbours
            self.neighbour.append((point_array - np.array([1, 0, 0])).tolist())  # -x
            self.neighbour.append((point_array + np.array([1, 0, 0])).tolist())  # +x
            self.neighbour.append((point_array - np.array([0, 1, 0])).tolist())  # -y
            self.neighbour.append((point_array + np.array([0, 1, 0])).tolist())  # +y
            self.neighbour.append((point_array - np.array([0, 0, 1])).tolist())  # -z
            self.neighbour.append((point_array + np.array([0, 0, 1])).tolist())  # +z
        else:
            raise ValueError('The input point should be a 3d vector')


# define 2D point class (seed) which has properties self and neighbour
class region_point_2D:
    def __init__(self, point):
        point_array = np.asarray(point)
        if point_array.ndim == 1 and len(point_array) == 2: # if point is a 2d point vector
            self.point = point
            self.neighbour = []    # generate neighbours
            self.neighbour.append((point_array - np.array([1, 0])).tolist())  # -x
            self.neighbour.append((point_array + np.array([1, 0])).tolist())  # +x
            self.neighbour.append((point_array - np.array([0, 1])).tolist())  # -y
            self.neighbour.append((point_array + np.array([0, 1])).tolist())  # +y
        else:
            raise ValueError('The input point should be a 2d vector')

class region_point_2D_vector:
    def __init__(self, point, origin):
        point_array = np.asarray(point)
        vec_origin = (point_array - np.asarray(origin)).tolist()
        vec_neighbours = [[1, 0], [-1, 0], [0, 1], [0, -1]]
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
    def __init__(self, point, dim = 2):
        if dim == 2:
            print('create 2D region')
            seed = region_point_2D(point)
        else:
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
    def update_final_boundary(self, dim=2):
        print('updating tooth surface')
        for point in self.region:
            count = 0
            if dim == 2:
                region_point_tem = region_point_2D(point)
            else:
                region_point_tem = region_point_3D(point)
            for neighbour_tem in region_point_tem.neighbour:
                if neighbour_tem in self.region:
                    count += 1
            if dim == 2:
                if count < 4:
                    self.boundary_final.append(point)
            if dim == 3:
                if count < 6:
                    self.boundary_final.append(point)

    # update final boundary test slice method
    def update_final_boundary_slice(self, slice):
        print('updating tooth surface')
        for point in self.region:
            point_check = point.copy()
            count = 0
            region_point_tem = region_point_2D(point)
            for neighbour_tem in region_point_tem.neighbour:
                if neighbour_tem in self.region:
                    count += 1
            if count < 4:
                #print('point_check is', point_check)
                #print('point_check_append is', point_check.append(2))
                #print('slice is', slice)
                point_check.append(slice)
                #print('point_check after append is', point_check)
                self.boundary_final.append(point_check)
        print('shape of boundary final is', np.shape(self.boundary_final))

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
        for point in self.boundary:
            if point not in self.region:
                self.region.append(point)


path_point_surface_matching = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\CT tooth segmentation\\CASE 5277\\CT3\\"
#path_point_surface_matching = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\HW-9898 Point-to-surface experiments\\CT scan\\drill_fxt_0.2\\"
# path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\FA-human scan\\'
scan, slice_properties = DR.load_scan_human(path_point_surface_matching)
fig = plt.figure()
#for i in range(503):
#    i = i+150
#    print('i is', i)
#    img = scan[i].pixel_array
#    HU = img * slice_properties[-2] + slice_properties[-1]
#    plt.imshow(HU)
#    plt.show()

#HU = img * slice_properties[-2] + slice_properties[-1]

img = scan[250].pixel_array[230:600, 200:620]
#img = scan[260].pixel_array
HU = img * slice_properties[-2] + slice_properties[-1]
test_img = np.zeros((HU.shape[0], HU.shape[1]))

HU_data = HU.flatten()
img_data = img.flatten()
HU_data_square = np.square(HU_data)
#y, bins, p = plt.hist(img_data, bins=20, density=True, alpha=0.6, color='g')
y, bins, p = plt.hist(HU_data, bins=200, density=True, alpha=0.6, color='g')


fig = plt.figure()
plt.imshow(HU)
plt.show()


dimension = 2
initial_seed = [260, 337]

if dimension == 2:
    SEED = initial_seed
    depth = 250
    region_candidate = region(SEED, dim = dimension)
    srg(region_candidate, scan, depth, dim=dimension)
    region_candidate.update_final_boundary(dim=dimension)

    for point in region_candidate.boundary_final:
        HU[point[0], point[1]] = 3000
    plt.imshow(HU)

if dimension == 3:
    plot_points = []
    print('checking code')
    i = 0
    for depth in range(270, 300, 1):
        print('depth is', depth)
        if i == 0:
            seed = initial_seed
            print('seed is', seed)
        else:
            seed = [int(b) for b in seed_candidate[0:2]]
            print('seed is', seed)

        region_candidate = region(seed, dim=2)
        srg(region_candidate, scan, depth, dim=2)
        region_candidate.update_final_boundary_slice(depth)
        plot_points.append(region_candidate.boundary_final)
        seed_candidate = np.mean(region_candidate.boundary_final, axis=0)
        #print('seed_candidate is', seed_candidate[0:2])

        img = scan[depth].pixel_array[230:600, 200:620]
        HU = img * slice_properties[-2] + slice_properties[-1]
        #plt.imshow(HU)
        #plt.show()
        for point in region_candidate.boundary_final:
            HU[point[0], point[1]] = 3000
        #plt.imshow(HU)
        #plt.show()

        del region_candidate
        i += 1
        #region_candidate.clear_final_boundary()
        #fig = plt.figure()
        #for point in region_candidate.boundary_final:
        #    HU[point[0], point[1]] = 3000
        #plt.imshow(HU)
        #plt.show()

    test_points = np.zeros(3)
    for i in range(len(plot_points)):
        test_points = np.vstack((test_points, np.asarray(plot_points[i])))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #print('shape of boundary is', np.shape(region_candidate.boundary_final))
    #Yomiplot.plot_3d_points(region_candidate.boundary_final, ax, axe_option=False)
    print('shape of boundary is', np.shape(test_points))
    Yomiplot.plot_3d_points(test_points[1:,:], ax, color='green', alpha=0.3, axe_option=False)


if dimension == 4:
    seed = [323, 360, 285]
    dimension = dimension - 1
    region_candidate = region(seed, dim = dimension)
    srg_3d(region_candidate, scan, dim=dimension)
    region_candidate.update_final_boundary(dim=dimension)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    print('shape of boundary is', np.shape(region_candidate.boundary_final))
    Yomiplot.plot_3d_points(region_candidate.boundary_final, ax, color='green', alpha=0.3, axe_option=False)

#fig2 = plt.figure()

plt.show()


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

