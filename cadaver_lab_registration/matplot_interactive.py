import numpy as np
import matplotlib.pyplot as plt
import DICOM_reader as DR
import array_processing as ap
from bitarray import bitarray
from bitarray import util as bitarray_util

import array_processing as ap
import coordinates

# import slider bar in matplot
from matplotlib.widgets import Slider, Button

# import helper functions for srg 3d segmentation
#import srg_3d_bitwise as srg3d

import point_cloud_manipulation as pcm
import plot as Yomiplot

# draw a circle in figure.
def draw_circle(point1, point2):
    center = point1
    radius = np.linalg.norm(np.asarray(point1) - np.asarray(point2))
    tolerance = 0.9
    xmin = np.int(center[0] - radius)
    xmax = np.int(center[0] + radius)
    ymin = np.int(center[1] - radius)
    ymax = np.int(center[1] + radius)
    circle = []
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            dis_tem = np.linalg.norm(np.asarray(point1) - np.asarray((x,y)))
            if np.abs(dis_tem - radius) < tolerance:
                circle.append((x,y))
    return circle, radius

def draw_line(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    xmin = np.min([x1, x2])
    xmax = np.max([x1, x2])
    line = []
    for x in range(xmin, xmax):
        y = (y2 - y1)/(x2 - x1) * (x - x1) + y1
        line.append((int(x), int(y)))
    return line


# ct img class
class ct_img:
    def __init__(self, ct_file_path):
        self.file_path = ct_file_path
        self.scan, self.slice_properties = DR.load_scan_human(self.file_path)
        self.fig, self.ax = plt.subplots()

        self.pixel_array = None
        self.pixel_array_tem = None
        self.click_points = []
        self.line_tem = []
        self.input_flag = True
        self.ix = 0
        self.iy = 0

        # SRG method
        self.seed = []
        self.search_radius = []
        self.seed_selection_figure = None

        # quadrate points method
        self.corner_list = []
        for i in range(32):
            self.corner_list.append([])
        self.corner_points = []
        self.box_selection_figure = None

    def update(self, val):
        self.ax.clear()
        self.ax.imshow(self.scan[np.int(val)].pixel_array)
        # fig.canvas.draw()
        # fig.canvas.draw_idle()

    def plot_ct_slices(self):
        init_slice = 35
        img = self.scan[init_slice].pixel_array
        HU = img * self.slice_properties[-2] + self.slice_properties[-1]
        plt.imshow(HU)

        self.ax.set_xlabel('Slice')
        axcolor = 'lightgoldenrodyellow'
        self.ax.margins(x=0)
        plt.subplots_adjust(left=0.25, bottom=0.25)

        # Make a horizontal oriented slider to control the slice
        axslice = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        slice_slider = Slider(
            ax=axslice,
            label="Slice",
            valmin=0,
            valmax=self.slice_properties[2],
            valinit=init_slice
        )
        slice_slider.on_changed(self.update)
        plt.show()

    # SRG method select seed
    def select_seed(self, slice_number, figure_control):
        img = self.scan[slice_number].pixel_array
        HU = img * self.slice_properties[-2] + self.slice_properties[-1]
        self.pixel_array = HU
        self.pixel_array_tem = np.copy(self.pixel_array)

        self.seed.clear()
        self.search_radius.clear()
        self.click_points.clear()
        self.seed_selection_figure = figure_control
        plt.imshow(self.pixel_array)
        print('Select seed points and radius in figure')
        self.seed_selection_figure.canvas.mpl_connect('button_press_event', self.onclick_srg)
        plt.show()

    # SRG method onclick event
    def onclick_srg(self, event):
        print('event is', event)
        self.ix, self.iy = event.xdata, event.ydata
        print('x = %d, y = %d' % (self.ix, self.iy))
        self.click_points.append((np.int(self.ix), np.int(self.iy)))
        if len(self.click_points) % 2 == 0:
            self.seed.clear()
            self.search_radius.clear()
            for i in range(len(self.click_points) // 2):
                circle_points, radius = draw_circle(self.click_points[i * 2], self.click_points[i * 2 + 1])
                self.seed.append(self.click_points[i * 2])
                self.search_radius.append(radius)
                for point in circle_points:
                    self.pixel_array_tem[point[1], point[0]] = 3000
        plt.imshow(self.pixel_array_tem)
        self.seed_selection_figure.canvas.draw()

    def clear_figure(self):
        self.fig.clear()
        plt.close(self.fig)

    def read_ct(self, slice_number):
        img = self.scan[slice_number].pixel_array
        HU = img * self.slice_properties[-2] + self.slice_properties[-1]
        self.pixel_array = HU
        self.pixel_array_tem = np.copy(self.pixel_array)

    # box method select corners
    def select_box(self, figure_control):
        #img = self.scan[slice_number].pixel_array
        #HU = img * self.slice_properties[-2] + self.slice_properties[-1]
        self.click_points.clear()
        self.line_tem.clear()
        #self.pixel_array = HU
        #self.pixel_array_tem = np.copy(self.pixel_array)
        self.box_selection_figure = figure_control
        plt.imshow(self.pixel_array_tem)
        print('Select box corners in the figure')
        self.box_selection_figure.canvas.mpl_connect('button_press_event', self.onclick_box)
        plt.show()

    # box method onclick event
    def onclick_box(self, event):
        print('event is', event)
        self.ix, self.iy = event.xdata, event.ydata
        print('x = %d, y = %d' % (self.ix, self.iy))
        point_tem = (np.int(self.ix), np.int(self.iy))
        self.click_points.append(point_tem)
        if len(self.click_points) == 4:
            self.corner_points.clear()
            for i in range(4):
                if i < 3:
                    line = draw_line(self.click_points[i], self.click_points[i+1])
                    self.line_tem.append(line)
                    for point in line:
                        self.pixel_array_tem[point[1], point[0]] = 3000
                else:
                    line = draw_line(self.click_points[3], self.click_points[0])
                    self.line_tem.append(line)
                    for point in line:
                        self.pixel_array_tem[point[1], point[0]] = 3000
            self.corner_points = self.click_points.copy()
        elif len(self.click_points) > 4:
            print('length of corner points is', len(self.corner_points))
            print('length of line_tem is', len(self.line_tem))
            # update the point in box
            point_idx = self.update_closed_point(point_tem)
            # find the old line idx
            line_idx_1 = point_idx - 1
            if line_idx_1 < 0:
                line_idx_1 = 3
            line_idx_2 = point_idx
            line_idx_3 = point_idx + 1
            if line_idx_3 > 3:
                line_idx_3 = 0

            print('line_idx1 is', line_idx_1)
            print('line_idx2 is', line_idx_2)
            print('line_idx3 is', line_idx_3)
            # reset the old line color
            for point in self.line_tem[line_idx_1]:
                self.pixel_array_tem[point[1], point[0]] = self.pixel_array[point[1], point[0]]
            for point in self.line_tem[line_idx_2]:
                self.pixel_array_tem[point[1], point[0]] = self.pixel_array[point[1], point[0]]
            # update the new line list and draw color
            line_1 = draw_line(self.corner_points[line_idx_1], self.corner_points[line_idx_2])
            line_2 = draw_line(self.corner_points[line_idx_2], self.corner_points[line_idx_3])
            self.line_tem[line_idx_1] = line_1
            self.line_tem[line_idx_2] = line_2
            for point in self.line_tem[line_idx_1]:
                self.pixel_array_tem[point[1], point[0]] = 3000
            for point in self.line_tem[line_idx_2]:
                self.pixel_array_tem[point[1], point[0]] = 3000

        print('self.corner points are', self.corner_points)
        plt.imshow(self.pixel_array_tem)
        self.box_selection_figure.canvas.draw()

    # Box method update closed point
    def update_closed_point(self, point_tem):
        if self.corner_points == []:
            print('nothing to update')
            min_dis_idx = 0
        else:
            dis = []
            for point in self.corner_points:
                print('point is', point)
                print('point_tem is', point_tem)
                dis.append(np.linalg.norm(np.asarray(point_tem) - np.asarray(point)))
            min_dis = min(dis)
            min_dis_idx = dis.index(min_dis)
            self.corner_points[min_dis_idx] = point_tem
        return min_dis_idx

    # Box method update selected box points list
    def update_box_list(self):
        self.corner_list.append(self.corner_points)


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

    # update final boundary
    def update_final_boundary_bitwise(self, z_threshold):
        print('updating tooth surface')
        print('remove duplicated points')
        self.remove_region_duplicate()
        print('check boundary')
        for point in self.region:
            count_tem = check_POINT_NEIGHBOUR_COUNT(point)
            if count_tem < 6:
            #if count_tem < 6 and point[2] > z_threshold:
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


# function to get intensity
def get_intensity(scan, coordinate): # coordinate is real [x, y, z]
    z_tem = coordinate[2]
    y_tem = coordinate[1]
    x_tem = coordinate[0]
    intensity = scan[z_tem].pixel_array[y_tem, x_tem]
    return intensity


# function to get intensity values from CT scan
def intensity_3d(ct_file, point_location):
    x = point_location[0]
    y = point_location[1]
    z = point_location[2]
    img = ct_file.scan[z].pixel_array
    HU = img * ct_file.slice_properties[-2] + ct_file.slice_properties[-1]
    intensity_value = HU[y, x]
    #print('intensity_value is', intensity_value)
    return intensity_value


# Convert 3d matrix idx to 1d array
# note: 3d matrix is in
# append each row
def get_array_idx_3d_1d(idx_3d):
    size_z = MATRIX_SIZE[0] # the number of the 2d matrices in 3d matrix (shape[0])
    size_y = MATRIX_SIZE[1] # row number of the 2d array
    size_x = MATRIX_SIZE[2] # column number of the 2d array
    idx_3d_x = idx_3d[0]
    idx_3d_y = idx_3d[1]
    idx_3d_z = idx_3d[2]
    idx_1d = idx_3d_z * size_y * size_x + idx_3d_y * size_x + idx_3d_x
    return idx_1d


# check point flag
# 1:  already checked in segmentation
# 0:  not checked in segmentation
def check_point_segmentation_label(point):
    idx_1d = get_array_idx_3d_1d(point)
    return POINT_LABEL[idx_1d]


# set point flag
# 1:  set to 1 after checking the point
def set_point_segmentation_label(point):
    idx_1d = get_array_idx_3d_1d(point)
    POINT_LABEL[idx_1d] = 1


# check point neighbour count flag
def check_POINT_NEIGHBOUR_COUNT(point):
    idx_1d = get_array_idx_3d_1d(point)
    bit0 = POINT_NEIGHBOUR_COUNT_0[idx_1d]
    bit1 = POINT_NEIGHBOUR_COUNT_1[idx_1d]
    bit2 = POINT_NEIGHBOUR_COUNT_2[idx_1d]
    count = bit0 * 4 + bit1 * 2 + bit2
    return count


# update point neighbour count flag
def update_POINT_NEIGHBOUR_COUNT(point):
    neighbour_points = generate_neighbour_point_list(point)
    for point_tem in neighbour_points:
        idx_1d = get_array_idx_3d_1d(point_tem)
        bit0 = POINT_NEIGHBOUR_COUNT_0[idx_1d]
        bit1 = POINT_NEIGHBOUR_COUNT_1[idx_1d]
        bit2 = POINT_NEIGHBOUR_COUNT_2[idx_1d]
        #print('bit in list_bitarray before is', bit0, bit1, bit2)
        count = bit0 * 4 + bit1 * 2 + bit2
        count += 1
        bit_new = bitarray_util.int2ba(count)
        #print('bit_new is', bit_new)
        bit_new_len = len(bit_new)
        if bit_new_len > 2:
            POINT_NEIGHBOUR_COUNT_0[idx_1d] = bit_new[-3]
            POINT_NEIGHBOUR_COUNT_1[idx_1d] = bit_new[-2]
            POINT_NEIGHBOUR_COUNT_2[idx_1d] = bit_new[-1]
        elif bit_new_len == 2:
            POINT_NEIGHBOUR_COUNT_0[idx_1d] = bit0
            POINT_NEIGHBOUR_COUNT_1[idx_1d] = bit_new[-2]
            POINT_NEIGHBOUR_COUNT_2[idx_1d] = bit_new[-1]
        else:
            POINT_NEIGHBOUR_COUNT_0[idx_1d] = bit0
            POINT_NEIGHBOUR_COUNT_1[idx_1d] = bit1
            POINT_NEIGHBOUR_COUNT_2[idx_1d] = bit_new[-1]
        #print('bit in list_bitarray after is',POINT_NEIGHBOUR_COUNT_0[idx_1d], POINT_NEIGHBOUR_COUNT_1[idx_1d], POINT_NEIGHBOUR_COUNT_2[idx_1d])
        del count, bit_new


# generate neighbour point list
def generate_neighbour_point_list(point):
    point_array = np.asarray(point)
    vec_neighbours = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    neighbour = []
    for vec in vec_neighbours:
        neighbour.append((point_array + np.asarray(vec)).tolist())
    return neighbour


# SRG method
def srg_3d_bitwise(region_candidate, ct_file, z_threshold):
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
            #if check_point_segmentation_label(point) == 0:
            region_candidate.region.append(point)
            #if i < 1:
            region_point_tem = region_point_3D(point)
            #else:
            #    region_point_tem = region_point_3D_vector(point, SEED)

            #check all neighbours, if they are not in the region already, check if they are qualified for tooth.
            for neighbour_point_tem in region_point_tem.neighbour:
                #idx_tem = get_array_idx_3d_1d(neighbour_point_tem)
                #print('neighbour_point_tem z is', neighbour_point_tem[2])

                # intensity_diff = intensity_3d(ct_file, neighbour_point_tem) - intensity_3d(ct_file, point)
                # if intensity_diff > 0:  # if intensity is increasing, count the point
                #     region_candidate.boundary_candidate.append(neighbour_point_tem)  # add to boundary_candidate
                #     set_point_segmentation_label(neighbour_point_tem)
                #     update_POINT_NEIGHBOUR_COUNT(neighbour_point_tem)
                # elif intensity_diff > CUTOFF:
                #     if intensity_3d(ct_file, neighbour_point_tem) > 600:  # if intensity is qualified
                #         region_candidate.boundary_candidate.append(neighbour_point_tem)  # add to boundary_candidate
                #         set_point_segmentation_label(neighbour_point_tem)
                #         update_POINT_NEIGHBOUR_COUNT(neighbour_point_tem)

                if neighbour_point_tem[2] > z_threshold[0] - 1 and neighbour_point_tem[2] < z_threshold[1] + 1: # if the point is above the z_threshold
                    # if neighbour_point_tem not in region_candidate.region:  # if it's not in region already
                    if check_point_segmentation_label(neighbour_point_tem) == 0:
                        if intensity_3d(ct_file, neighbour_point_tem) > 700:   # if intensity is qualified
                            region_candidate.boundary_candidate.append(neighbour_point_tem) # add to boundary_candidate
                            set_point_segmentation_label(neighbour_point_tem)
                            update_POINT_NEIGHBOUR_COUNT(neighbour_point_tem)

        if len(region_candidate.boundary_candidate) < 1:  # if no qualified points found, terminate SRG
            is_srg_finished = True

        if i == SRG_LOOPS:
            is_srg_finished = True

        del region_point_tem
        i += 1

    return True



if __name__ == '__main__':
    #SEEDS = []
    path_point_surface_matching = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\" \
                                  "CT tooth segmentation\\CASE 5277\\CT3\\"
    #path_point_surface_matching = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\HW-9898 Point-to-surface experiments\\CT scan\\drill_fxt_0.2\\"
    # path = 'G:\\My Drive\\Project\\IntraOral Scanner Registration\\FA-human scan\\'

    # read ct images and check each slice to define the depth boundary for tooth segmentation
    test = ct_img(path_point_surface_matching)
    test.plot_ct_slices()

    # Define z_boundaries
    z_min = np.int(input("Enter the value of minimum z \n"))
    z_max = np.int(input("Enter the value of maximum z \n"))
    selected_slice = z_min
    print('z_min is', z_min)
    print('z_max is', z_max)

    test.read_ct(selected_slice)

    start_tooth_segmentation = True
    while start_tooth_segmentation is True:
        fig1 = plt.figure()
        test.select_box(fig1)
        input_flag = True
        while input_flag is True:
            val = input("Press 'p' to proceed or 'r' to repeat data collection procedure \n")
            if val == 'r':
                fig1 = plt.figure()
                test.select_box(selected_slice, fig1)
            elif val == 'p':
                print('selected box is', test.corner_points)
                val = np.int(input("Enter the tooth number"))
                write_box_corners = True
                while write_box_corners is True:
                    if -1 < val < 33:
                        test.corner_list[val-1] = test.corner_points.copy()
                        write_box_corners = False
                    else:
                        val = np.int(input("Wrong number was entered, please enter a number between 0-32"))
                input_flag = False
            else:
                print('Wrong input is provided \n')
        print('corner list is', test.corner_list)

        ask_if_continue = True
        while ask_if_continue is True:
            val = int(input("Continue to segment another tooth? \n press '1' to continue, press '0' to stop \n"))
            if val == 1:
                print('continue to segment the next tooth')
                ask_if_continue = False
            elif val == 0:
                print('segmentation is finished')
                start_tooth_segmentation = False
                ask_if_continue = False
            else:
                print('wrong value was entered \n')



    fig1 = plt.figure()
    test.select_seed(selected_slice, fig1)
    input_flag = True
    while input_flag is True:
        val = input("Press 'p' to proceed or 'r' to repeat data collection procedure \n")
        if val == 'r':
            fig1 = plt.figure()
            print('shape of center pc is', len(test.seed))
            test.select_seed(selected_slice, fig1)
        elif val == 'p':
            print('shape of center pc is', len(test.seed))
            input_flag = False
        else:
            print('Wrong input is provided \n')

    test.clear_figure()

    # Tooth segmentation using SRG
    mesh_z = test.slice_properties[2]
    mesh_x = test.slice_properties[4]  # x is defined in real x direction (space)
    mesh_y = test.slice_properties[3]  # y is defined in real y direction (space)
    MATRIX_SIZE = [mesh_z, mesh_y, mesh_x]

    POINT_LABEL = bitarray(mesh_z * mesh_y * mesh_x)
    POINT_LABEL.setall(0)

    POINT_NEIGHBOUR_COUNT_0 = bitarray(mesh_z * mesh_y * mesh_x)
    POINT_NEIGHBOUR_COUNT_1 = bitarray(mesh_z * mesh_y * mesh_x)
    POINT_NEIGHBOUR_COUNT_2 = bitarray(mesh_z * mesh_y * mesh_x)
    POINT_NEIGHBOUR_COUNT_0.setall(0)
    POINT_NEIGHBOUR_COUNT_1.setall(0)
    POINT_NEIGHBOUR_COUNT_2.setall(0)

    SEED = [test.seed[0][0], test.seed[0][1], z_min]  # tooth 21
    # SEED = [288, 445, 265]  # tooth 22
    # SEED = [490, 540, 260] # tooth 17
    # SEED = [474, 542, 280]
    # [323, 360, 285]
    z_tho = [z_min-1, z_max]
    SRG_LOOPS = 80
    CUTOFF = -200
    DISTANCE_THO = test.search_radius
    print('SEED is', SEED)
    print('DISTANCE_THO is', DISTANCE_THO)

    region_candidate = region(SEED)
    time1 = np.datetime64('now')
    srg_3d_bitwise(region_candidate, test, z_threshold=z_tho)
    # srg_3d(region_candidate, scan, z_threshold=z_tho)
    time2 = np.datetime64('now')
    print('segmentation time is', time2 - time1)

    # region_candidate.update_final_boundary(z_threshold=z_tho)
    region_candidate.update_final_boundary_bitwise(z_threshold=z_tho)
    time3 = np.datetime64('now')
    print('finding boundary time is', time3 - time2)
    print('size of boundary points are', np.shape(region_candidate.boundary_final))

    pc = pcm.preprocess_point_cloud(region_candidate.boundary_final, voxel_size=4.0)
    print('shape of pc is', np.shape(pc))
    # source = o3d.PointCloud()
    # source.points = o3d.Vector3dVector(region_candidate.boundary_final)
    # source.paint_uniform_color([0, 0.651, 0.929])  # yellow
    # o3d.draw_geometries([source])

    plt.figure()
    ax = plt.axes(projection='3d')
    print('shape of boundary is', np.shape(region_candidate.boundary_final))
    Yomiplot.plot_3d_points(pc, ax, color='green', alpha=0.3, axe_option=False)

    plt.show()
