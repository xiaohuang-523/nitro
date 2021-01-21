import numpy as np


def unpack(kp_2b_update, p, kp_update_parameters):
    # the np.copy() used here is to solve a known issue regarding pointers. please leave as it for now.
    kp = np.copy(kp_2b_update)
    phat = np.copy(p)
    kp_fit = kp_update_parameters[:]
    xx = np.nonzero(kp_fit)
    kp[xx] = phat
    k = kp[:]
    return k


def pack(kp_2b_packed, kp_update_parameters):
    # the np.copy() used here is to solve a known issue regarding pointers. please leave as it for now.
    kp = np.copy(kp_2b_packed)
    kp_p = np.copy(kp_update_parameters)
    p_tem = kp[np.nonzero(kp_p)]
    p = np.array(p_tem)
    return p


def dup_bl(bl, measurements):
    m, n = np.shape(measurements)
    bl_tem = bl
    bl_add = bl
    for i in range(m):
        bl_tem = np.vstack((bl_tem, bl_add))
    bl_f = bl_tem[1:]
    return bl_f
