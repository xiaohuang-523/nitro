# Developing point-surface matching algorithm
# Modified ICP using SVD point registration
# Key point is to select points from know surfaces
# by Xiao Huang @ 09/10/2021

import numpy as np
import Readers as Yomiread
import plot as Yomiplot
import matplotlib.pyplot as plt
import array_processing as ap
import registration

import open3d as o3d

def point_surface_distance(point, surface):
    dis_abs = np.linalg.norm(point - surface[0,:])
    dis_vec = point - surface[0,:]
    selected_point = surface[0,:]
    raw_dis = []
    for point_tem in surface:
        dis_tem = np.linalg.norm(point - point_tem)
        raw_dis.append(dis_tem)
        if dis_tem < dis_abs:
            dis_abs = dis_tem
            dis_vec = point - point_tem
            del selected_point
            selected_point = point_tem
        #print('dis_tem is', dis_tem)
        #print('dis_abs is', dis_abs)
    return dis_abs, dis_vec, selected_point, raw_dis


def points_tooth_surface_distance(point_list, surface, direction_idx):
    tooth_surface_distance = []
    for point in point_list:
        dis_abs, dis_vec, point_tem = point_surface_distance(point, surface)
        if dis_vec[direction_idx] < 0:
            dis = -dis_abs
        else:
            dis = dis_abs
        tooth_surface_distance.append(dis)
    return tooth_surface_distance


def select_closest_point(point_list, surface):
    selected_points = []
    for point in point_list:
        dis_abs, dis_vec, point_tem, raw_dis = point_surface_distance(point, surface)
        selected_points.append(point_tem)
    return np.asarray(selected_points)


def select_closest_point_on_tooth(probing_points_transformed, buccal, front, lingual, occlusal):
    buccal_selected_points = select_closest_point(probing_points_transformed[0:3,:], buccal)
    front_selected_points = select_closest_point(probing_points_transformed[3:6,:], front)
    lingual_selected_points = select_closest_point(probing_points_transformed[6:9,:], lingual)
    occlusal_selected_points = select_closest_point(probing_points_transformed[9:12,:], occlusal)
    return buccal_selected_points, front_selected_points


def point_surface_distance_direction_search(point, surface, direction_idx, tolerance):
    dis = 1.0
    dis_vec = np.zeros(3)
    for point_tem in surface:
        plane_dis = []
        # select points in the same cylinder
        for i in range(3):
            if i != direction_idx:
                plane_dis.append(np.abs(point[i] - point_tem[i]))
        if all(element < tolerance for element in plane_dis):
            dis_tem = (point - point_tem)[direction_idx]
            if np.abs(dis_tem) < np.abs(dis):
                dis = dis_tem
                dis_vec = point - point_tem
        del plane_dis
    return dis, dis_vec


def points_tooth_surface_distance_direction_search(point_list, surface, direction_idx, tolerance):
    tooth_surface_distance = []
    for point in point_list:
        dis, dis_vec = point_surface_distance_direction_search(point, surface, direction_idx, tolerance)
        tooth_surface_distance.append(dis)
    return tooth_surface_distance


def solve_translation(probing_points_transformed, buccal, front, occlusal):
    # buccal surface distances ( dy>0: outside, dy<0: inside)
    buccal_distance = points_tooth_surface_distance_direction_search(probing_points_transformed[0:3, :], buccal, 1, 0.4)
    print('buccal_distance is', buccal_distance)

    # front surface distances ( dx>0: outside, dx<0: inside)
    front_distance = points_tooth_surface_distance_direction_search(probing_points_transformed[3:6, :], front, 0, 0.4)
    print('front_distance is', front_distance)

    # occlusal surface distances (dz<0: outside, dz>0: inside)
    occlusal_distance = points_tooth_surface_distance_direction_search(probing_points_transformed[9:12, :], occlusal, 2, 0.4)
    print('occlusal_distance is', occlusal_distance)

    # generate transformation
    dx = -np.sum(np.asarray(front_distance)) / 3
    dy = -np.sum(np.asarray(buccal_distance)) / 3
    dz = -np.sum(np.asarray(occlusal_distance)) / 3
    print('dx is', dx)
    print('dy is', dy)
    print('dz is', dz)

    #probing_points_transformed = probing_points_transformed + np.asarray([dx, dy, dz])
    return np.asarray([dx, dy, dz])


def solve_rotation(probing_points_transformed, buccal, front, occlusal):
    # buccal surface distances ( dy>0: outside, dy<0: inside)
    buccal_distance = points_tooth_surface_distance(probing_points_transformed[0:3, :], buccal, 1)
    print('buccal_distance is', buccal_distance)

    # front surface distances ( dx>0: outside, dx<0: inside)
    front_distance = points_tooth_surface_distance(probing_points_transformed[3:6, :], front, 0)
    print('front_distance is', front_distance)

    # occlusal surface distances (dz<0: outside, dz>0: inside)
    occlusal_distance = points_tooth_surface_distance(probing_points_transformed[9:12, :], occlusal, 2)
    print('occlusal_distance is', occlusal_distance)

    # generate transformation
    dx = -np.sum(np.asarray(front_distance)) / 3
    dy = -np.sum(np.asarray(buccal_distance)) / 3
    dz = -np.sum(np.asarray(occlusal_distance)) / 3
    print('dx is', dx)
    print('dy is', dy)
    print('dz is', dz)

    #probing_points_transformed = probing_points_transformed + np.asarray([dx, dy, dz])
    return np.asarray([dx, dy, dz])


def modified_ICP(source_points, buccal, front, lingual, occlusal):
    probing_points_transformed = source_points
    buccal_selected_points = select_closest_point(probing_points_transformed[0:3, :], buccal)
    front_selected_points = select_closest_point(probing_points_transformed[3:6, :], front)
    lingual_selected_points = select_closest_point(probing_points_transformed[9:12, :], lingual)
    occlusal_selected_points = select_closest_point(probing_points_transformed[6:9, :], occlusal)
    destination_points = np.ones(3)
    destination_points = np.vstack((destination_points, buccal_selected_points))
    destination_points = np.vstack((destination_points, front_selected_points))
    destination_points = np.vstack((destination_points, lingual_selected_points))
    destination_points = np.vstack((destination_points, occlusal_selected_points))
    destination_points = np.asarray(destination_points)[1:, :]
    #print('ios_points_transformed_2 check are', source_points)
    r_, t_ = registration.point_set_registration(source_points, destination_points)
    #print('r is', r_)
    #print('t is', t_)
    ios_points_transformed_2 = np.ones(3)
    for point in source_points:
        #print('point is', point)
        #print('t_ is', t_)
        point_tem_2 = np.matmul(r_, point) + t_
        #print('point_tem_2 is', point_tem_2)
        ios_points_transformed_2 = np.vstack((ios_points_transformed_2, point_tem_2))
    ios_points_transformed_2 = ios_points_transformed_2[1:, :]
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #Yomiplot.plot_3d_points(ios_points_transformed_2,ax,color='red')
    #Yomiplot.plot_3d_points(destination_points, ax, color='green')
    #Yomiplot.plot_3d_points(source_points, ax, color='blue')
    #Yomiplot.plot_3d_points(occlusal,ax,color='red', alpha=0.2)
    #plt.show()


    #print('ios_points_transformed_2 are', ios_points_transformed_2)
    error = np.linalg.norm(ios_points_transformed_2 - destination_points, axis=1)
    #print('error is', error)
    error = np.sqrt(np.sum(error**2)/len(error))
    print('error rmse is', error)
    return error, ios_points_transformed_2, destination_points


# read surface points
CT_FILE_BASE = "G:\\My Drive\\Project\\HW-9232 Registration method for edentulous\\point-surface registration\\"
j = 31
occlusal = Yomiread.read_csv(CT_FILE_BASE + "dicom_points_tooth" + np.str(j) + "_occlusal.csv", 3, -1)
buccal = Yomiread.read_csv(CT_FILE_BASE + "dicom_points_tooth" + np.str(j) + "_buccal.csv", 3, -1)
lingual = Yomiread.read_csv(CT_FILE_BASE + "dicom_points_tooth" + np.str(j) + "_lingual.csv", 3, -1)
front = Yomiread.read_csv(CT_FILE_BASE + "dicom_points_tooth" + np.str(j) + "_front.csv", 3, -1)

surface = np.zeros(3)
surface = np.vstack((surface, occlusal))
surface = np.vstack((surface, buccal))
surface = np.vstack((surface, lingual))
surface = np.vstack((surface, front))
surface = surface[1:,:]

volume_centroid = np.sum(surface, axis=0)/np.shape(surface)[0]
occlusal_surface_centroid = np.sum(occlusal, axis=0)/np.shape(occlusal)[0]
buccal_surface_centroid = np.sum(buccal, axis=0)/np.shape(buccal)[0]
lingual_surface_centroid = np.sum(lingual, axis=0)/np.shape(lingual)[0]
front_surface_centroid = np.sum(front, axis=0)/np.shape(front)[0]
surface_centroids = ap.combine_elements_in_list([buccal_surface_centroid, front_surface_centroid, occlusal_surface_centroid, lingual_surface_centroid])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Yomiplot.plot_3d_points(surface, ax, color='green')
plt.show()

# read IOS points
# p1-p3: buccal side (positive direction: +y(CT) )
# p4-p6: front side  (positive direction: +x(CT) )
# p7-p9: occlusal side (positive direction: -z(CT) )
# p10-p12: lingual side
ios_points = Yomiread.read_csv(CT_FILE_BASE + "points_raw.txt", 4, -1, delimiter=' ')[:,1:]
buccal_point_center = np.sum(ios_points[0:3,:], axis=0)/3
front_point_center = np.sum(ios_points[3:6,:], axis=0)/3
occlusal_point_center = np.sum(ios_points[6:9,:], axis=0)/3
lingual_point_center = np.sum(ios_points[9:12,:], axis=0)/3
point_centers = ap.combine_elements_in_list([buccal_point_center, front_point_center, occlusal_point_center, lingual_point_center])
print(ios_points)

# Initial alignment
R, t = registration.point_set_registration(point_centers, surface_centroids)
ios_points_transformed = []
for point in ios_points:
    ios_points_transformed.append(np.matmul(R, point) + t)
ios_points_transformed = np.asarray(ios_points_transformed)
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
Yomiplot.plot_3d_points(buccal, ax, color='green')
Yomiplot.plot_3d_points(ios_points_transformed, ax, color='red')
plt.show()

# Modified ICP
source_points = ios_points_transformed
for k in range(50):
    error, ios_points_transformed_new, destination_points_new = modified_ICP(source_points,buccal,front,lingual,occlusal)
    source_points = ios_points_transformed_new
    #print('destination_points are', destination_points_new[0,:])
    if error < 0.01:
        break
fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
Yomiplot.plot_3d_points(occlusal, ax, color='green', alpha=0.2)
Yomiplot.plot_3d_points(ios_points_transformed, ax, color='red')
Yomiplot.plot_3d_points(source_points, ax, color='blue')
plt.show()

exit()

# prepare destination points
probing_points_transformed = ios_points_transformed
buccal_selected_points = select_closest_point(probing_points_transformed[0:3, :], buccal)
front_selected_points = select_closest_point(probing_points_transformed[3:6, :], front)
lingual_selected_points = select_closest_point(probing_points_transformed[6:9, :], lingual)
occlusal_selected_points = select_closest_point(probing_points_transformed[9:12, :], occlusal)
destination_points = np.ones(3)
destination_points = np.vstack((destination_points, buccal_selected_points))
destination_points = np.vstack((destination_points, front_selected_points))
destination_points = np.vstack((destination_points, lingual_selected_points))
destination_points = np.vstack((destination_points, occlusal_selected_points))
destination_points = np.asarray(destination_points)[1:,:]

r_, t_ = registration.point_set_registration(ios_points_transformed, destination_points)
ios_points_transformed_2 = np.ones(3)
for point in ios_points_transformed:
    point_tem_2 = np.matmul(r_, point) + t_
    ios_points_transformed_2 = np.vstack((ios_points_transformed_2, point_tem_2))
ios_points_transformed_2 = ios_points_transformed_2[1:,:]
error2 = np.linalg.norm(ios_points_transformed_2 - destination_points, axis=1)
print('error 2 is', error2)
exit()


# point to surface matching -- distance based -----------------------------------------------------------------------
# solve translation
t = np.zeros(3)

# align x
#for k in range(1):
#while(True):
for j in range(50):
    print('j is', j)
        #t_tem = solve_translation(ios_points_transformed, buccal, front, occlusal)
        #ios_points_transformed = ios_points_transformed + np.array([0, t_tem[0], 0])  # move in x first
        # adjust y
    for i in range(50):
        print('i is', i)
        t_tem = solve_translation(ios_points_transformed, buccal, front, occlusal)
        if np.abs(t_tem[1]) < 0.2:
                # if all(np.abs(element) < 0.2 for element in t_tem)
            break
        t = t + t_tem
        ios_points_transformed = ios_points_transformed + np.array([0, t_tem[1], 0])

        # adjust x
    if np.abs(t_tem[0]) < 0.2:
        break
    ios_points_transformed = ios_points_transformed + np.array([t_tem[0], 0, 0])


    #if np.abs(t_tem[2]) < 0.2:
    #    break
    #ios_points_transformed = ios_points_transformed + np.array([0, 0, t_tem[2]])
    ## align y
    #for i in range(50):
    #    print('i is', i)
    #    t_tem = solve_translation(ios_points_transformed, buccal, front, occlusal)
    #    if np.abs(t_tem[1]) < 0.2:
        #if all(np.abs(element) < 0.2 for element in t_tem):
    #        break
    #    t = t + t_tem
    #    ios_points_transformed = ios_points_transformed + np.array([0, t_tem[1], 0])





# solve rotation
fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
Yomiplot.plot_3d_points(surface, ax, color='green')
Yomiplot.plot_3d_points(ios_points_transformed, ax, color='red')
plt.show()
