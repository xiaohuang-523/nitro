# Edentulous_arch class for point-pair registration
# by Xiao Huang @ 07/29/2021
import numpy as np
import coordinates
import array_processing as ap


# dynamic grouping
# point is in the format [r, theta, z]
# there are only three regions in radius (inner, occlusal and outer)
#       inner and outer:  z_regions * theta_regions
#       occlusal:   theta_regions
#
# result is region_number which starts from 1
def divide_region( radius_range, angle_range, z_range, D_RADIUS, D_ANGLE, D_HEIGHT, point):
    delta_angle = (angle_range[1] - angle_range[0]) / D_ANGLE
    delta_height = (z_range[1] - z_range[0]) / D_HEIGHT
    delta_radius = (radius_range[1] - radius_range[0]) / D_RADIUS
    angle_array = []
    for i in range(D_ANGLE):
        angle_array.append(angle_range[0] + i * delta_angle)
    angle_array.append(angle_range[1])
    radius_array = []
    for i in range(D_RADIUS):
        radius_array.append(radius_range[0] + i * delta_radius)
    radius_array.append(radius_range[1])
    height_array = []
    for i in range(D_HEIGHT):
        height_array.append(z_range[0] + i * delta_height)
    height_array.append(z_range[1])

    # check with region does the point belong to
    r_idx = ap.check_interval_idx_single_value(point[0], radius_array)
    theta_idx = ap.check_interval_idx_single_value(point[1]*180/np.pi, angle_array)
    z_idx = ap.check_interval_idx_single_value(point[2], height_array)

    # due to the definition, r_region is always divided into three sub-regions (inner, occlusal, outer)
    # which gives
    #           r_idx = 0:   inner
    #           r_idx = 1:   outer
    #           r_idx = 2:   occlusal
    #
    # theta_region and height_region are defined differently
    #           theta_idx = 0:  outside
    #           theta_idx > D_ANGLE: outside
    #           theta_idx:   the 'theta_idx'th region in theta regions. I.E., theta_idx = 2, the 2nd theta region

    region_number = -1 # means point is outside of the check region

    if r_idx == 0:  # inner
        if 0 < theta_idx <= D_ANGLE and 0 < z_idx <= D_HEIGHT:
            region_number = (theta_idx-1) * D_HEIGHT + z_idx
    elif r_idx == 1:  # occlusal
        if 0 < theta_idx <= D_ANGLE:
            region_number = D_ANGLE * D_HEIGHT + theta_idx
    else: # outer
        if 0 < theta_idx <= D_ANGLE and 0 < z_idx <= D_HEIGHT:
            region_number = D_ANGLE * D_HEIGHT + D_ANGLE + (theta_idx - 1) * D_HEIGHT + z_idx

    return region_number


# Edentulous arch class for registration experiments
# Class components:
#       1. All surface points
#       2. Coordinate center
#       3. Cartesian: three region surface points (outer, inner, occlusal)
#       4. Cylindrical: three region surface points (outer, inner, occlusal)
#       5. Target origins (4 center, axes are along the Z)
#
# Note
#       1. Target should be in ascending order
#
class Edentulous_arch:
    def __init__(self, stl_raw_points, defined_targets, defined_center=[0, 0, 0]):
        self.all_points_cartesian = np.asarray(stl_raw_points)
        self.coordinate_center = defined_center

        # convert to cylindrical coordinates [r, theta, z]
        self.all_points_cylindrical = coordinates.convert_cylindrical(self.all_points_cartesian, defined_center)
        self.angle_range = [np.min(self.all_points_cylindrical[:, 1] * 180 / np.pi),
                            np.max(self.all_points_cylindrical[:, 1] * 180 / np.pi)]
        self.radius_range = [np.min(self.all_points_cylindrical[:, 0]), np.max(self.all_points_cylindrical[:, 0])]
        self.height_range = [np.min(self.all_points_cylindrical[:, 2]), np.max(self.all_points_cylindrical[:, 2])]

        # define targets
        self.target_origins_cartesian = np.asarray(defined_targets)
        self.target_origins_cylindrical = coordinates.convert_cylindrical(self.target_origins_cartesian, defined_center)

        # define multiple regions based on cylindrical coordinate angles (theta)
        if np.ndim(self.target_origins_cylindrical) == 1:
            self.target_angle_range = self.angle_range
        else:
            self.target_angle_range = []
            self.target_angle_range.append(self.angle_range[0])
            for i in range(len(self.target_origins_cylindrical[:, 1]) - 1):
                self.target_angle_range.append((self.target_origins_cylindrical[i, 1]
                                            + self.target_origins_cylindrical[i+1, 1])/2 * 180/np.pi)
            self.target_angle_range.append(self.angle_range[1])

        self.target_radius_range = self.radius_range
        self.target_height_range = self.height_range
        
        self.all_regions_cartesion = []
        self.all_regions_cylindrical = []
        
        self.check_angle_range = []
        self.check_radius_range = []
        self.check_height_range = []
        self.n_fiducial = 0
        self.n_target = len(self.target_origins_cartesian)
        self.default_fiducial = []

    # divide arch into different regions based on settings
    # Arguments:
    #   1. D_RADIUS, D_ANGLE, D_HEIGHT:  the number of regions in each direction. By default, D_radius is 3 as the space
    #       is generally divided into three regions by default (inner, occlusal, outer)
    #   2. defined_radius_range: The maximum and minimum values in the corresponding directions
    #   3. check_for_target flag:
    #       Define if the separation is done for each of the targets or for the entire arch.
    #       check_for_target = 0:  check for the entire arch.
    #       check_for_target = 1:  check for each individual target. In this case, the variables defined above are
    #                              for each target.
    def divide_arch(self, D_RADIUS, D_ANGLE, D_HEIGHT, 
                    defined_radius_range = [], defined_angle_range = [], defined_height_range = [], 
                    check_for_target = 0):
        print('Perform single target simulation')
        if len(defined_radius_range) == 0:
            self.check_radius_range = self.target_radius_range
        else:
            self.check_radius_range = defined_radius_range

        if check_for_target == 0:
            if len(defined_angle_range) == 0:
                self.check_angle_range = self.angle_range
            else:
                self.check_angle_range = defined_angle_range
        elif check_for_target == 1:
            if len(defined_angle_range) == 0:
                self.check_angle_range = self.target_angle_range
            else:
                self.check_angle_range = []
                self.check_angle_range.append(defined_angle_range[0])
                for i in range(len(self.target_origins_cylindrical[:, 1]) - 1):
                    self.check_angle_range.append((self.target_origins_cylindrical[i, 1]
                                                    + self.target_origins_cylindrical[i + 1, 1]) / 2 * 180/np.pi)
                self.check_angle_range.append(defined_angle_range[1])
        
        if len(defined_height_range) == 0:
            self.check_height_range = self.target_height_range
        else:
            self.check_height_range = defined_height_range
            
        if check_for_target == 0:
            self.n_fiducial = 2 * D_ANGLE * D_HEIGHT + D_ANGLE
            print('total number of fiducials are', self.n_fiducial)
            for i in range(self.n_fiducial):
                self.all_regions_cartesion.append([])
                self.all_regions_cylindrical.append([])

            for i in range(len(self.all_points_cylindrical)):
                x = divide_region(self.check_radius_range, self.check_angle_range, self.check_height_range, D_RADIUS, D_ANGLE, D_HEIGHT, 
                                  self.all_points_cylindrical[i, :])
                if x != -1:
                    self.all_regions_cartesion[x - 1].append(self.all_points_cartesian[i, :])
                    self.all_regions_cylindrical[x - 1].append(self.all_points_cylindrical[i, :])
                    
        if check_for_target == 1:
            self.n_fiducial = self.n_target * (2 * D_ANGLE * D_HEIGHT + D_ANGLE)
            print('total number of targets is', self.n_target)
            print('total number of fiducials is', self.n_fiducial)
            print('fiducial number for each target is', self.n_fiducial / self.n_target)
            for i in range(self.n_fiducial):
                self.all_regions_cartesion.append([])
                self.all_regions_cylindrical.append([])

            for i in range(self.n_target):
                check_angle_range = [self.check_angle_range[i], self.check_angle_range[i+1]]
                for j in range(len(self.all_points_cylindrical)):
                    x = divide_region(self.check_radius_range, check_angle_range, self.check_height_range, D_RADIUS,
                                  D_ANGLE, D_HEIGHT,
                                  self.all_points_cylindrical[j, :])
                    if x != -1:
                        x = x + i * (2 * D_ANGLE * D_HEIGHT + D_ANGLE)
                        self.all_regions_cartesion[x - 1].append(self.all_points_cartesian[j, :])
                        self.all_regions_cylindrical[x - 1].append(self.all_points_cylindrical[j, :])

        for region in self.all_regions_cartesion:
            print('region shape is', np.shape(region))
            center = np.sum(np.asarray(region), axis=0) / len(region)
            self.default_fiducial.append(center)
        self.default_fiducial = np.asarray(self.default_fiducial)

    # divide arch into different regions based on settings
    # Arguments:
    #   1. D_RADIUS, D_ANGLE, D_HEIGHT:  the number of regions in each direction. By default, D_radius is 3 as the space
    #       is generally divided into three regions by default (inner, occlusal, outer)
    #   2. individual_defined_range
    #       for each target range, the radius and height ranges are defined separately.
    #       the angle range is defined in the edentulous_arch class.
    #       individual defined ragnes are defined as
    #               [ [target 1 list], [target 2 list], ... , ]
    #               where
    #               [target 1 list] is [[radius range], [height range]]
    def divide_arch_individual_target(self, D_RADIUS, D_ANGLE, D_HEIGHT, defined_angle_range, individual_defined_range):
        print('Perform multiple target simulation, configuration I')
        self.n_fiducial = self.n_target * (2 * D_ANGLE * D_HEIGHT + D_ANGLE)
        print('total number of targets is', self.n_target)
        print('total number of fiducials is', self.n_fiducial)
        print('fiducial number for each target is', self.n_fiducial / self.n_target)
        for i in range(self.n_fiducial):
            self.all_regions_cartesion.append([])
            self.all_regions_cylindrical.append([])

        self.check_angle_range = []
        self.check_angle_range.append(defined_angle_range[0])
        for i in range(len(self.target_origins_cylindrical[:, 1]) - 1):
            self.check_angle_range.append((self.target_origins_cylindrical[i, 1]
                                           + self.target_origins_cylindrical[i + 1, 1]) / 2 * 180 / np.pi)
        self.check_angle_range.append(defined_angle_range[1])
        print('check_angle_range is', self.check_angle_range)

        for i in range(self.n_target):
            check_radius_range = individual_defined_range[i][0]
            check_height_range = individual_defined_range[i][1]
            check_angle_range = [self.check_angle_range[i], self.check_angle_range[i + 1]]
            for j in range(len(self.all_points_cylindrical)):
                x = divide_region(check_radius_range, check_angle_range, check_height_range, D_RADIUS, D_ANGLE,
                                  D_HEIGHT, self.all_points_cylindrical[j, :])
                if x != -1:
                    x = x + i * (2 * D_ANGLE * D_HEIGHT + D_ANGLE)
                    self.all_regions_cartesion[x - 1].append(self.all_points_cartesian[j, :])
                    self.all_regions_cylindrical[x - 1].append(self.all_points_cylindrical[j, :])

        for region in self.all_regions_cartesion:
            print('region shape is', np.shape(region))
            center = np.sum(np.asarray(region), axis=0) / len(region)
            self.default_fiducial.append(center)
        self.default_fiducial = np.asarray(self.default_fiducial)

    # divide arch into different regions based on settings
    # Arguments:
    #   1. D_RADIUS, D_ANGLE, D_HEIGHT:  the number of regions in each direction. By default, D_radius is 3 as the space
    #       is generally divided into three regions by default (inner, occlusal, outer)
    #   2. individual_defined_range
    #       for each target range, the radius, angle and height ranges are defined separately.
    #       individual defined ranges are defined as
    #               [ [target 1 list], [target 2 list], ... , ]
    #               where
    #               [target 1 list] is [[radius range], [height range], [angle span]]
    def divide_arch_individual_target_modified_angle(self, D_RADIUS, D_ANGLE, D_HEIGHT, individual_defined_range):
        print('Perform multiple target simulation, configuration II')
        self.n_fiducial = self.n_target * (2 * D_ANGLE * D_HEIGHT + D_ANGLE)
        print('total number of targets is', self.n_target)
        print('total number of fiducials is', self.n_fiducial)
        print('fiducial number for each target is', self.n_fiducial / self.n_target)
        for i in range(self.n_fiducial):
            self.all_regions_cartesion.append([])
            self.all_regions_cylindrical.append([])

        angle_center = []
        for i in range(len(self.target_origins_cylindrical[:, 1])):
            angle_center.append(self.target_origins_cylindrical[i, 1] * 180 / np.pi)
        #print('angle_center is', angle_center)

        for i in range(self.n_target):
            check_radius_range = individual_defined_range[i][0]
            check_height_range = individual_defined_range[i][1]
            check_angle_range = [angle_center[i] - individual_defined_range[i][2]/2,
                                 angle_center[i] + individual_defined_range[i][2]/2]
            print('check_angle_range is', check_angle_range)
            for j in range(len(self.all_points_cylindrical)):
                x = divide_region(check_radius_range, check_angle_range, check_height_range, D_RADIUS, D_ANGLE,
                                  D_HEIGHT, self.all_points_cylindrical[j, :])
                if x != -1:
                    x = x + i * (2 * D_ANGLE * D_HEIGHT + D_ANGLE)
                    self.all_regions_cartesion[x - 1].append(self.all_points_cartesian[j, :])
                    self.all_regions_cylindrical[x - 1].append(self.all_points_cylindrical[j, :])

        for region in self.all_regions_cartesion:
            print('region shape is', np.shape(region))
            center = np.sum(np.asarray(region), axis=0) / len(region)
            self.default_fiducial.append(center)
        self.default_fiducial = np.asarray(self.default_fiducial)

    # divide arch into different regions based on settings
    # Arguments:
    #   1. D_RADIUS, D_ANGLE, D_HEIGHT:  the number of regions in each direction. By default, D_radius is 3 as the space
    #       is generally divided into three regions by default (inner, occlusal, outer)
    #   2. individual_defined_range
    #       for each target range, the radius, angle and height ranges are defined separately.
    #       individual defined ranges are defined as
    #               [ [target 1 list], [target 2 list], ... , ]
    #               where
    #               [target 1 list] is [[radius range], [height range], [angle span]]
    #   3. location_offset
    #       in application, the location of selected regions may be offset by an angle.
    def divide_arch_individual_target_modified_angle_with_offset(self, D_RADIUS, D_ANGLE, D_HEIGHT, individual_defined_range, location_offset):
        print('Perform multiple target simulation, configuration II with offset')
        self.n_fiducial = self.n_target * (2 * D_ANGLE * D_HEIGHT + D_ANGLE)
        print('total number of targets is', self.n_target)
        print('total number of fiducials is', self.n_fiducial)
        print('fiducial number for each target is', self.n_fiducial / self.n_target)
        for i in range(self.n_fiducial):
            self.all_regions_cartesion.append([])
            self.all_regions_cylindrical.append([])

        angle_center = []
        for i in range(len(self.target_origins_cylindrical[:, 1])):
            angle_center.append(self.target_origins_cylindrical[i, 1] * 180 / np.pi)

        for i in range(self.n_target):
            check_radius_range = individual_defined_range[i][0]
            check_height_range = individual_defined_range[i][1]
            check_angle_range = [angle_center[i] - individual_defined_range[i][2]/2 + location_offset,
                                 angle_center[i] + individual_defined_range[i][2]/2+location_offset]
            print('check_angle_range is', check_angle_range)
            for j in range(len(self.all_points_cylindrical)):
                x = divide_region(check_radius_range, check_angle_range, check_height_range, D_RADIUS, D_ANGLE,
                                  D_HEIGHT, self.all_points_cylindrical[j, :])
                if x != -1:
                    x = x + i * (2 * D_ANGLE * D_HEIGHT + D_ANGLE)
                    self.all_regions_cartesion[x - 1].append(self.all_points_cartesian[j, :])
                    self.all_regions_cylindrical[x - 1].append(self.all_points_cylindrical[j, :])

        for region in self.all_regions_cartesion:
            print('region shape is', np.shape(region))
            center = np.sum(np.asarray(region), axis=0) / len(region)
            self.default_fiducial.append(center)
        self.default_fiducial = np.asarray(self.default_fiducial)

    # divide arch into different regions based on settings
    # Arguments:
    #   1. D_RADIUS, D_ANGLE, D_HEIGHT:  the number of regions in each direction. By default, D_radius is 3 as the space
    #       is generally divided into three regions by default (inner, occlusal, outer)
    #   2. individual_defined_range
    #       for each target range, the radius, angle and height ranges are defined separately.
    #       individual defined ranges are defined as
    #               [ [target 1 list], [target 2 list], ... , ]
    #               where
    #               [target 1 list] is [[radius range], [height range], [angle span]]
    def divide_arch_individual_target_modified_angle_seg(self, D_RADIUS, D_ANGLE, D_HEIGHT, individual_defined_range, angle_span):
        print('Perform multiple target simulation, configuration III')
        self.n_fiducial = (self.n_target+1) * (2 * D_ANGLE * D_HEIGHT + D_ANGLE)
        print('total number of targets is', self.n_target)
        print('total number of fiducials is', self.n_fiducial)
        print('fiducial number for each target is', self.n_fiducial / (self.n_target+1))
        for i in range(self.n_fiducial):
            self.all_regions_cartesion.append([])
            self.all_regions_cylindrical.append([])

        check_angle_range = []
        check_angle_range.append(self.target_origins_cylindrical[0,1] * 180 / np.pi - 20)
        for i in range(len(self.target_origins_cylindrical[:, 1]) - 1):
            check_angle_range.append((self.target_origins_cylindrical[i, 1]
                                           + self.target_origins_cylindrical[i + 1, 1]) / 2 * 180 / np.pi)
        check_angle_range.append(self.target_origins_cylindrical[-1,1] * 180 / np.pi + 20)
        print('check_angle_range is', check_angle_range)

        for i in range(self.n_target + 1):
            check_radius_range = individual_defined_range[i][0]
            check_height_range = individual_defined_range[i][1]
            check_angle_range_tem = [check_angle_range[i] - angle_span/2, check_angle_range[i] + angle_span/2]
            print('check_angle_range_tem is', check_angle_range_tem)
            for j in range(len(self.all_points_cylindrical)):
                x = divide_region(check_radius_range, check_angle_range_tem, check_height_range, D_RADIUS, D_ANGLE,
                                  D_HEIGHT, self.all_points_cylindrical[j, :])
                if x != -1:
                    x = x + i * (2 * D_ANGLE * D_HEIGHT + D_ANGLE)
                    self.all_regions_cartesion[x - 1].append(self.all_points_cartesian[j, :])
                    self.all_regions_cylindrical[x - 1].append(self.all_points_cylindrical[j, :])

        for region in self.all_regions_cartesion:
            print('region shape is', np.shape(region))
            center = np.sum(np.asarray(region), axis=0) / len(region)
            self.default_fiducial.append(center)
        self.default_fiducial = np.asarray(self.default_fiducial)


