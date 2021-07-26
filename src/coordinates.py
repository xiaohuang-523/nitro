import numpy as np

# Convert to cylindrical coordinates
def convert_cylindrical(pc, center):
    if np.ndim(pc) == 1:
        dx = pc[0] - center[0]
        dy = pc[1] - center[1]
        angle = np.arctan2(dy, dx)
        if angle < -np.pi / 2:
            angle = angle + 2 * np.pi
        theta = angle
        r = np.sqrt(dx ** 2 + dy ** 2)
        z = pc[2]

    else:
        theta = []
        r = []
        z = []
        for point in pc:
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            angle = np.arctan2(dy, dx)
            if angle < -np.pi/2:
                angle = angle + 2*np.pi
            theta.append(angle)
            r.append(np.sqrt(dx**2 + dy**2))
            z.append(point[2])
    pc_cylindrical = np.asarray([r, theta, z])
    return pc_cylindrical.transpose()


# Convert cylindrical coordinates to cartesian
# pc is [r, theta, z]
def convert_cartesian(pc, center):
    x = []
    y = []
    #z = []
    for point in pc:
        dx = point[0] * np.cos(point[1])
        dy = point[0] * np.sin(point[1])
        x.append(center[0] + dx)
        y.append(center[1] + dy)
        #z.append(point[2])
    #pc_cylindrical = np.asarray([x, y, z])

    pc_cartesian = np.asarray([x, y])
    print('pc_cartesian is', pc_cartesian)
    return pc_cartesian.transpose()


# Generate coordinate system
def generate_frame(p1, p2, p3):
    x = (p3 - p2) / np.linalg.norm(p3-p2)
    z = np.cross(x, (p1-p2)/np.linalg.norm(p1-p2))
    y = np.cross(z, x)
    o = p1
    T = np.eye(4)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = o
    return np.linalg.inv(T)


# Generate coordinate system in Yomi convention
def generate_frame_yomi(p1, p2, p3):
    y = (p3 - p2) / np.linalg.norm(p3-p2)
    z = np.cross((p1-p3), y)/np.linalg.norm(np.cross(p1-p3, y))
    x = np.cross(y, z)
    o = p1
    T = np.eye(4)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = o
    T_inv_check = np.eye(4)
    T_inv_check[0:3,0:3] = np.transpose(T[0:3,0:3])
    T_inv_check[0:3,3] = -o

    return np.linalg.inv(T)
