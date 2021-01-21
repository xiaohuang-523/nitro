import numpy as np
import functools
import math

# Introduction on use
# Three FK functions are defined in this file
#   1. FW_Kinematics_Matrices
#      Solve the FK in Yomi convention. J0 value will be automatically added.
#      E.g.,
#      If the KDL include 7 links where the first link is common base link (link 0, Link1, ..., Link6),
#      then the input joints in this function is J1, J2, J3, J4, J5, J6
#      where J0 will be added automatically.
#
#   2. FW_Kinematics_Matrices_no_common
#      Solve the FK in Yomi convention, J0 value should be added before use if common base is considered.
#      E.g.,
#      If the KDL include 7 links: Link1, Link2, ..., Link7,
#      then the input joints is J1, j2, ..., J7
#
#   3. FW_Kinematics_DH_classical
#      Solve the FK in classical DH convention.



# Solve the homo transformation matrix in Yomi convention
def Yomi_Base_Matrix(p):
    # solve the known issues related to the pointers
    p_tem = np.copy(p)
    # The base matrix should be in the order of Tx, Ty, Tz, Rz, Ry, Rx
    Tx = p_tem[0] ; Ty = p_tem[1] ; Tz = p_tem[2] ; Rx = p_tem[5] ; Ry = p_tem[4] ; Rz = p_tem[3]
    Matrix=np.asarray([
        [np.cos(Ry)*np.cos(Rz),     np.cos(Rz)*np.sin(Rx)*np.sin(Ry)-np.cos(Rx)*np.sin(Rz),     np.cos(Rx)*np.cos(Rz)*np.sin(Ry)+np.sin(Rx)*np.sin(Rz),\
                                                        Tx*np.cos(Ry)*np.cos(Rz)+ Ty*(np.cos(Rz)*np.sin(Rx)*np.sin(Ry)- np.cos(Rx)*np.sin(Rz))+ Tz*(np.cos(Rx)*np.cos(Rz)*np.sin(Ry)+np.sin(Rx)*np.sin(Rz))],\
        [np.cos(Ry)*np.sin(Rz),     np.cos(Rx)*np.cos(Rz)+np.sin(Ry)*np.sin(Rx)*np.sin(Rz),     -np.cos(Rz)*np.sin(Rx)+np.cos(Rx)*np.sin(Ry)*np.sin(Rz),\
                                                        Tx*np.cos(Ry)*np.sin(Rz)+ Ty*(np.cos(Rz)*np.cos(Rx)+ np.sin(Rz)*np.sin(Ry)*np.sin(Rx))+ Tz*(-np.cos(Rz)*np.sin(Rx)+np.cos(Rx)*np.sin(Ry)*np.sin(Rz))],\
        [-np.sin(Ry),               np.cos(Ry)*np.sin(Rx),                                      np.cos(Rx)*np.cos(Ry),
                                                        Tz*np.cos(Rx)*np.cos(Ry)+ Ty*np.cos(Ry)*np.sin(Rx)-Tx*np.sin(Ry)],
        [0,0,0,1]])
    return Matrix


# Convert DH parameters (classical) to Yomi convention
def DH_2_Yomi_classical(p):
    p_tem0 = np.copy(p)
    Yomi_parameters = np.ones(6)
    for m in range(int(len(p_tem0)/4)):
        p_tem = p_tem0[4*m:4*(m+1)]
        # DH parameters are stored in the order of alpha, theta, a, d
        alpha = p_tem[0];   theta = p_tem[1];   a = p_tem[2];   d = p_tem[3]
        Ry = 0
        Rz = theta
        Rx = alpha
        #   Tz * np.cos(Rx) + Ty * np.sin(Rx) = d
        #   Tx * np.sin(Rz) + Ty * np.cos(Rz) * np.cos(Rx) + Tz * (-np.cos(Rz) * np.sin(Rx)) = a * sin(theta)
        #   Tx * np.cos(Rz) + Ty * (-np.cos(Rx) * np.sin(Rz)) + Tz * np.sin(Rx) * np.sin(Rz) = a * cos(theta)
        A = np.array([
            [0., np.sin(Rx), np.cos(Rx)],
            [np.sin(Rz), np.cos(Rz)*np.cos(Rx), -np.cos(Rz)*np.sin(Rx)],
            [np.cos(Rz), -np.cos(Rx)*np.sin(Rz), np.sin(Rx)*np.sin(Rz)]
        ])
        b = np.array([d, a*np.sin(theta), a*np.cos(theta)])
        T = np.matmul(np.linalg.inv(A), b)
        Yomi_parameters_tem = np.array([T[0], T[1], T[2], Rz, Ry, Rx])
        Yomi_parameters = np.hstack((Yomi_parameters,Yomi_parameters_tem))
    return Yomi_parameters[6:]


def DH_Matrix_modified(p):
    # solve the known issues related to the pointers
    p_tem = np.copy(p)
    # The base matrix should be in the order of alpha, theta, a, d
    alpha = p_tem[0] ; theta = p_tem[1] ; a = p_tem[2] ; d = p_tem[3];
    # Version 2: Rx*Tx*Rz*Tz
    # For more information, check http://jntuhceh.org/web/tutorials/faculty/873_ic29-2014.pdf
    Matrix=np.asarray([[np.cos(theta), -np.sin(theta), 0, a],
                       [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
                       [np.sin(alpha)*np.sin(theta), np.sin(alpha)*np.cos(theta), np.cos(alpha), d*np.cos(alpha)],
                       [0,0,0,1]])
    return Matrix


def DH_Matrix_classical(p):
    # solve the known issues related to the pointers
    p_tem = np.copy(p)
    # The base matrix should be in the order of alpha, theta, a, d
    alpha = p_tem[0] ; theta = p_tem[1] ; a = p_tem[2] ; d = p_tem[3];
    # Version 1: Rz*Tz*Tx*Rx
    # For information, check http://jntuhceh.org/web/tutorials/faculty/873_ic29-2014.pdf
    Matrix=np.asarray([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                      [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.sin(alpha)*np.cos(theta), a*np.sin(theta)],
                      [0, np.sin(alpha), np.cos(alpha), d],
                      [0,0,0,1]])
    return Matrix


# In this function, the input joints are J1, J2, J3, ...
# J0 = 0 will be automatically added
# For inputs, the # of joints angles is 1 less than # of links in kdl.
def FW_Kinematics_Matrices(cal,q):
    # solve the known issues related to the pointers
    cal_tem = np.copy(cal)
    q_tem = np.copy(q)

    cal2 = np.reshape(cal_tem, (int(len(cal_tem)/6), 6))
    MatrixList1 = [] ; MatrixList2 = []
    for x in range(len(cal2) - 1):
        cal2[x + 1][3] += q_tem[x]
    for x in cal2:
        MatrixList1.append(Yomi_Base_Matrix(x))
    for x in range(len(MatrixList1)):
        MatrixX = functools.reduce(np.matmul, MatrixList1[0:x + 1])
        MatrixList2.append(MatrixX)
    return(MatrixList2)


# This function does not add J0 automatically.
# User should mannually add J0=0 if the common base link is also considered in the kdl.
# For inputs, the # of joint angles is the same with # of links in kdl.
def FW_Kinematics_Matrices_no_common(cal,q):
    # solve the known issues related to the pointers
    cal_tem = np.copy(cal)
    q_tem = np.copy(q)
    cal2 = np.reshape(cal_tem, (int(len(cal_tem)/6), 6))
    MatrixList1 = [] ; MatrixList2 = []
    for x in range(len(cal2)):
        cal2[x][3] += q_tem[x]
    for x in cal2:
        MatrixList1.append(Yomi_Base_Matrix(x))
    for x in range(len(MatrixList1)):
        MatrixX = functools.reduce(np.matmul, MatrixList1[0:x + 1])
        MatrixList2.append(MatrixX)
    return(MatrixList2)


# Forward kinematics in DH convention. (Classical DH)
def FW_Kinematics_DH_classical(cal,q):
    # solve the known issues related to the pointers
    cal_tem = np.copy(cal)
    q_tem = np.copy(q)
    cal2 = np.reshape(cal_tem, (int(len(cal_tem)/4), 4))
    MatrixList1 = [] ; MatrixList2 = []
    for x in range(len(cal2)):
        cal2[x][1] += q_tem[x]
    for x in cal2:
        MatrixList1.append(DH_Matrix_classical(x))
    for x in range(len(MatrixList1)):
        MatrixX = functools.reduce(np.matmul, MatrixList1[0:x + 1])
        MatrixList2.append(MatrixX)
    return(MatrixList2)


# Convert rotation matrix to quaternion
# Reference: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
def rot2quat(rot):
    if len(np.shape(rot)) != 2:
        print('error, function only supports 3D rotation matrix')
        exit()
    tr = np.trace(rot)
    if tr > 0:
        S = np.sqrt( tr + 1.0 ) * 2
        qw = 0.25 * S
        qx = (rot[2,1] - rot[1,2]) / S
        qy = (rot[0,2] - rot[2,0]) / S
        qz = (rot[1,0] - rot[0,1]) / S
    elif rot[0,0] > rot[1,1] and rot[0,0] > rot[2,2]:
        S = np.sqrt(1.0 + rot[0,0] - rot[1,1] - rot[2,2]) * 2
        qw = (rot[2,1] - rot[1,2]) / S
        qx = 0.25 * S
        qy = (rot[0,1] + rot[1,0]) / S
        qz = (rot[0,2] + rot[2,0]) / S
    elif rot[1,1] > rot[2,2]:
        S = np.sqrt(1.0 + rot[1,1] - rot[0,0] - rot[2,2]) * 2
        qw = (rot[0,2] - rot[2,0]) / S
        qx = (rot[0,1] + rot[1,0]) / S
        qy = 0.25 * S
        qz = (rot[1,2] + rot[2,1]) / S
    else:
        S = np.sqrt(1.0 + rot[2,2] - rot[0,0] - rot[1,1]) * 2
        qw = (rot[1,0] - rot[0,1]) / S
        qx = (rot[0,2] + rot[2,0]) / S
        qy = (rot[1,2] + rot[2,1]) / S
        qz = 0.25 * S
    return qw, qx, qy, qz


# Jacobian
# Geometric Jacobian method based on Chapter 3.1 in book "Robotics - Modelling, Planning and Control" by Bruno, Page 131
# For Schunk GuideArm, the common base transformation is needed in kdl. KDL has one more link than joint angles.
# I.E.  6X7 kdl + 6 joint angles
def solve_jacob(kdl, joint):
    joint_tem =np.insert(joint, 0, 0.)
    pos = FW_Kinematics_Matrices_no_common(kdl, joint_tem)
    joint_n = int(len(kdl) / 6)
    ee = pos[-1]
    jacobian = np.zeros((6,1))
    # Only calculate the positions of j1-6, which are pos[0] ---- pos[5]
    for i in range(joint_n-1):
        pos_tem = pos[i]
        jp_tem = np.cross(pos_tem[0:3, 2], ee[0:3,3] - pos_tem[0:3,3])
        jo_tem = pos_tem[0:3, 2]
        col_tem = np.reshape(np.append(jp_tem,jo_tem), (6,1))
        jacobian = np.hstack((jacobian, col_tem))
    return jacobian[:,1:]

# This code is not working
#def FW_Solve_DH(cal):
#    cal_tem = np.copy(cal)
#    cal2 = np.reshape(cal_tem, (int(len(cal_tem) / 6), 6))
#    kdl = np.array([1.,1.])
#    for x in cal2:
#        kdl = np.append(kdl, np.asarray(Solve_DH(x)))
#        print('cal2 is', x)
#        print('DH cal is', Solve_DH(x))
#        print('cal2 matrix is')
#        print(Yomi_Base_Matrix(x))
#        print('DH matrix is')
#        print(DH_Matrix(Solve_DH(x)))
    #MatrixList = np.asarray(MatrixList)
    #MatrixList = np.reshape(kdl, newshape=(28,1))
#    return kdl[2:]

# This code is not working. Testing the conversion from Yomi to DH
#def Solve_DH_2(p):
    # solve the known issues related to the pointers
#    p_tem = np.copy(p)
    # The base matrix should be in the order of Tx, Ty, Tz, Rz, Ry, Rx
#    Tx = p_tem[0] ; Ty = p_tem[1] ; Tz = p_tem[2] ; Rx = p_tem[5] ; Ry = p_tem[4] ; Rz = p_tem[3]
#    theta = np.arctan2(-np.cos(Rz)*np.sin(Rx)*np.sin(Ry)-np.cos(Rx)*np.sin(Rz),np.cos(Ry)*np.cos(Rz))
#    alpha = np.arctan2(-np.sin(Ry),np.cos(Ry)*np.sin(Rz))
#    a = Tx*np.cos(Ry)*np.cos(Rz)+ Ty*(np.cos(Rz)*np.sin(Rx)*np.sin(Ry)- np.cos(Rx)*np.sin(Rz))+ Tz*(np.cos(Rx)*np.cos(Rz)*np.sin(Ry)+np.sin(Rx)*np.sin(Rz))
#    if np.cos(alpha) == 0:
#        d = Tx*np.cos(Ry)*np.sin(Rz)+ Ty*(np.cos(Rz)*np.cos(Rx)+ np.sin(Rz)*np.sin(Ry)*np.sin(Rx))+ Tz*(-np.cos(Rz)*np.sin(Rx)+np.cos(Rx)*np.sin(Ry)*np.sin(Rz))/np.sin(alpha)
#    else:
#        d = Tz*np.cos(Rx)*np.cos(Ry)+ Ty*np.cos(Ry)*np.sin(Rx)-Tx*np.sin(Ry)/np.cos(alpha)
#    return alpha, theta, a, d

#def Solve_DH_1(p):
    # solve the known issues related to the pointers
#    p_tem = np.copy(p)
    # The base matrix should be in the order of Tx, Ty, Tz, Rz, Ry, Rx
#    Tx = p_tem[0] ; Ty = p_tem[1] ; Tz = p_tem[2] ; Rx = p_tem[5] ; Ry = p_tem[4] ; Rz = p_tem[3]
    #theta = Rz
#    theta = np.arctan2(np.cos(Ry)*np.sin(Rz),np.cos(Ry)*np.cos(Rz))
#    alpha = np.arctan2(np.sin(Rx),np.cos(Rx))
    # #print('alpha is', alpha)
#    d = Tz*np.cos(Rx)*np.cos(Ry)+ Ty*np.cos(Ry)*np.sin(Rx)-Tx*np.sin(Ry)
#    if np.sin(theta) == 0:
#        a = Tx*np.cos(Ry)*np.cos(Rz)+ Ty*(np.cos(Rz)*np.sin(Rx)*np.sin(Ry)- np.cos(Rx)*np.sin(Rz))+ Tz*(np.cos(Rx)*np.cos(Rz)*np.sin(Ry)+np.sin(Rx)*np.sin(Rz))/np.cos(Ry)*np.cos(Rz)
#    else:
#        a = Tx*np.cos(Ry)*np.sin(Rz)+ Ty*(np.cos(Rz)*np.cos(Rx)+ np.sin(Rz)*np.sin(Ry)*np.sin(Rx))+ Tz*(-np.cos(Rz)*np.sin(Rx)+np.cos(Rx)*np.sin(Ry)*np.sin(Rz))/np.cos(Ry)*np.sin(Rz)
#    return alpha, theta, a, d
