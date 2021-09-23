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
    for point_tem in surface:
        dis_tem = np.linalg.norm(point - point_tem)
        if dis_tem < dis_abs:
            dis_abs = dis_tem
            dis_vec = point - point_tem
            del selected_point
            selected_point = point_tem
    return dis_abs, dis_vec, selected_point


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
        dis_abs, dis_vec, point_tem = point_surface_distance(point, surface)
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


# solve rotation matrix based on Linear Least-Squares Optimization for Point-to-Plane ICP Surface Registration
def rotation_matrix(a, b, r):
    r11 = np.cos(r) * np.cos(b)
    r12 = -np.sin(r)*np.cos(a) + np.cos(r)*np.sin(b)*np.sin(a)
    r13 = np.sin(r)*np.sin(a) + np.cos(r)*np.sin(b)*np.cos(a)
    r21 = np.sin(r)*np.cos(b)
    r22 = np.cos(r)*np.cos(a) + np.sin(r)*np.sin(b)*np.sin(a)
    r23 = -np.cos(r)*np.sin(a) + np.sin(r)*np.sin(b)*np.cos(a)
    r31 = -np.sin(b)
    r32 = np.cos(b)*np.sin(a)
    r33 = np.cos(b)*np.cos(a)
    rot = np.array([[r11, r12, r13, 0],
                    [r21, r22, r23, 0],
                    [r31, r32, r33, 0],
                    [0, 0, 0, 1]])
    return rot


# solve minimization problem
def minimize_svd(source_points, destination_points):
    A = np.zeros(6)
    b = []
    for i in range(len(source_points)):
        source_tem = source_points[i, :]
        destination_tem = destination_points[i, :]
        A_tem = np.array([[0, source_tem[2] , -source_tem[1], 1, 0, 0],
                                   [-source_tem[2], 0, source_tem[0], 0, 1, 0],
                                   [source_tem[1], -source_tem[0], 0, 0, 0, 1]])
        b.append(destination_tem[0] - source_tem[0])
        b.append(destination_tem[1] - source_tem[1])
        b.append(destination_tem[2] - source_tem[2])
        A = np.vstack((A, A_tem))

    A = A[1:,:]
    b = np.asarray(b)
    A_p = np.linalg.pinv(A)     # pseudo-inverse of A
    x_opt = np.matmul(A_p, b)
    rot = rotation_matrix(x_opt[0], x_opt[1], x_opt[2])
    trans = np.array([[1, 0, 0, x_opt[3]],
                      [0, 1, 0, x_opt[4]],
                      [0, 0, 1, x_opt[5]],
                      [0, 0, 0, 1]])
    m_opt = np.matmul(trans, rot)
    destination_fit = []
    for point in source_points:
        point_tem = np.insert(point, 3, 1)
        destination_fit.append(np.matmul(m_opt, point_tem)[0:3])
    destination_fit = np.asarray(destination_fit)
    error = destination_fit - destination_points
    error = np.linalg.norm(error, axis=1)
    return destination_fit, error



def modified_ICP(source_points, buccal, front, lingual, occlusal):
    probing_points_transformed = source_points
    buccal_selected_points = select_closest_point(probing_points_transformed[0:3, :], buccal)
    front_selected_points = select_closest_point(probing_points_transformed[3:6, :], front)
    lingual_selected_points = select_closest_point(probing_points_transformed[6:9, :], lingual)
    occlusal_selected_points = select_closest_point(probing_points_transformed[9:12, :], occlusal)
    destination_points = np.ones(3)
    destination_points = np.vstack((destination_points, buccal_selected_points))
    destination_points = np.vstack((destination_points, front_selected_points))
    destination_points = np.vstack((destination_points, lingual_selected_points))
    destination_points = np.vstack((destination_points, occlusal_selected_points))
    destination_points = np.asarray(destination_points)[1:, :]

    new_s_points, error = minimize_svd(source_points, destination_points)
    #print('error is', error)
    error = np.sqrt(np.sum(error**2)/len(error))
    print('error rmse is', error)
    return error, new_s_points


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

source_pcd = o3d.PointCloud()
source_pcd.points = o3d.Vector3dVector(source_points)  # error, can not generate point cloud with 12 points.

target_pcd = o3d.PointCloud()
target_pcd.points = o3d.Vector3dVector(surface)

# Perform ICP registration
voxel_size = 1.0
threshold = 1.0
trans_init = np.eye(4)
# import ICP functions
import ICP
transform = ICP.registration_simple_pc(voxel_size, threshold, source_pcd, target_pcd, trans_init)




#for k in range(10):
#    error, ios_points_transformed_new= modified_ICP(source_points,buccal,front,lingual,occlusal)
#    source_points = ios_points_transformed_new
    #print('destination_points are', destination_points_new[0,:])
#    if error < 0.01:
#        break
#fig3 = plt.figure()
#ax = fig3.add_subplot(111, projection='3d')
#Yomiplot.plot_3d_points(front, ax, color='green', alpha=0.2)
#Yomiplot.plot_3d_points(ios_points_transformed, ax, color='red')
#Yomiplot.plot_3d_points(source_points, ax, color='blue')
#plt.show()

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
