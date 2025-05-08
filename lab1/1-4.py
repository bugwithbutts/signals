import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy 

from importlib.machinery import SourceFileLoader
module = SourceFileLoader("data.utils", 'lab1\\data files\\data\\utils.py').load_module()

from importlib.machinery import SourceFileLoader
module = SourceFileLoader("data.data", 'lab1\\data files\\data\\data.py').load_module()

C = np.array([[0.99376, -0.09722, 0.05466],
               [0.09971, 0.99401, -0.04475],
               [-0.04998, 0.04992, 0.9975]])
tt = np.array([[0.5, 0.1, 0.5]])
var_lidar = 1
var_gnss = 1
var_acc = 0.02
var_gyro = 0.02

def R(q):
    q0 = q[0]
    q1 = q[1][0]
    q2 = q[1][1]
    q3 = q[1][2]
    return np.array([[2 * q0 * q0 - 1 + 2 * q1 * q1, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
                      [2 * q1 * q2 + 2 * q0 * q3, 2 * q0 * q0 - 1 + 2 * q2 * q2, 2 * q2 * q3 - 2 * q0 * q1],
                      [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 2 * q0 * q0 - 1 + 2 * q3 * q3]])

def q(teta):
    norm = np.linalg.norm(teta)
    if norm == 0:
        return [1, np.array([0, 0, 0])]
    return [np.cos(norm / 2), np.array(teta / norm * np.sin(norm / 2))]

def mult(p, q):
    return [p[0] * q[0] - np.dot(p[1], q[1]), p[0] * q[1] + q[0] * p[1] + np.cross(p[1], q[1])]

def upd(state, dt, f, w):
    state[0] = state[0] + dt * state[1] + dt * dt / 2 * ((R(state[2]) @ np.array([f]).T).T + np.array([[0, 0, -9.81]]))
    state[1] = state[1] + dt * ((R(state[2]) @ np.array([f]).T).T + np.array([[0, 0, -9.81]]))
    state[2] = mult(state[2], q(w * dt))
    return state

def cross_matrix(a):
    return np.array([[0, -a[0, 2], a[0, 1]],
               [a[0, 2], 0, -a[0, 0]],
               [-a[0, 1], a[0, 0], 0]])

def F(dt, q, f):
    return np.block([[np.eye(3), np.eye(3) * dt, np.zeros((3, 3))],
                        [np.zeros((3, 3)), np.eye(3), -dt * cross_matrix((R(q) @ np.array([f]).T).T)],
                        [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)]])

def L():
    return np.block([[np.zeros((3, 3)), np.zeros((3, 3))],
                    [np.eye(3), np.zeros((3, 3))],
                    [np.zeros((3, 3)), np.eye(3)]])
def n(dt):
    return dt * dt * np.block([[var_acc * np.eye(3), np.zeros((3, 3))],
                                [np.zeros((3, 3)), var_gyro * np.eye(3)]])

def plus(mx, md):
    mx[0] = mx[0] + np.array([[md[0, 0], md[0, 1], md[0, 2]]])
    mx[1] = mx[1] + np.array([[md[0, 3], md[0, 4], md[0, 5]]])
    mx[2] = mult(q(np.array([md[0, 6], md[0, 7], md[0, 8]])), mx[2])
    return mx

def prediction(md, P, dt, q, f):
    # Prediction
    Fx = F(dt, q, f)
    P = Fx @ P @ Fx.T + L() @ n(dt) @ L().T
    return P

def kalman2(md, P, y, dt, q, f, p):
    Q = np.block([[np.eye(3) * var_gnss, np.zeros((3, 3))],
                  [np.zeros((3, 3)), np.eye(3) * var_lidar]])
    # Prediction
    P = prediction(md, P, dt, q, f)
    # Update
    Hx = np.block([[np.eye(3), np.zeros((3, 6))], 
                   [np.eye(3), np.zeros((3, 6))]])
    S = Hx @ P @ Hx.T + Q
    K = P @ Hx.T @ np.linalg.inv(S)
    md = (K @ (y - np.array([[p[0, 0], p[0, 1], p[0, 2], p[0, 0], p[0, 1], p[0, 2]]])).T).T
    P = P - K @ S @ K.T
    return md, P

def kalman1(md, P, y, dt, q, f, p, var):
    Q = np.eye(3) * var
    # Prediction
    P = prediction(md, P, dt, q, f)
    # Update
    Hx = np.hstack((np.eye(3), np.zeros((3, 6))))
    S = Hx @ P @ Hx.T + Q
    K = P @ Hx.T @ np.linalg.inv(S)
    md = (K @ (y - np.array([[p[0, 0], p[0, 1], p[0, 2]]])).T).T
    P = P - K @ S @ K.T
    return md, P

with open('D:\\everything\\code\\lab\\signals\\lab1\\data files\\data\\data.pkl', 'rb') as file:
    data = pickle.load(file)
    gt = data['gt']
    imu_f = data['imu_f']
    imu_w = data['imu_w']
    gnss = data['gnss']
    lidar = data['lidar']
    control, obs1, obs2 = [], [], []
    for i in range(1, len(imu_f.data)):
        control.append([imu_f.t[i], imu_f.data[i - 1], imu_w.data[i - 1]])
    for i in range(len(gnss.t)):
        obs1.append([gnss.t[i], np.array([gnss.data[i]])])
    for i in range(len(lidar.t)):
        obs2.append([lidar.t[i], (C @ lidar.data[i]).T + tt])
    mx = [np.array([gt.p[0]]), np.array([gt.v[0]]), [1, np.array([0, 0, 0])]]
    md = np.zeros((1, 9))
    P = np.eye(9) * 1
    lstx = 2.055
    filter_x, filter_y, filter_z = [], [], []
    for t, f, w in control:
        cur1 = -1
        cur2 = -1
        eps = 0.05
        for i in range(len(obs1)):
            if abs(t - obs1[i][0]) < eps:
                cur1 = i
                break
        for i in range(len(obs2)):
            if abs(t - obs2[i][0]) < eps:
                cur2 = i
                break
        qq = copy.deepcopy(mx[2])
        mx = upd(mx, t - lstx, f, w)
        pp = copy.deepcopy(mx[0])
        if cur1 != -1 and cur2 != -1: # Both observations are available
            md, P = kalman2(md, P, np.hstack((obs1[cur1][1], obs2[cur2][1])), t - lstx, qq, f, pp)
            mx = plus(mx, md)
        elif cur1 != -1: # GNSS is available
            md, P = kalman1(md, P, obs1[cur1][1], t - lstx, qq, f, pp, var_gnss)
            mx = plus(mx, md)
        elif cur2 != -1: # Lidar is available
            md, P = kalman1(md, P, obs2[cur2][1], t - lstx, qq, f, pp, var_lidar)
            mx = plus(mx, md)
        else: # No observations are available
            P = prediction(md, P, t - lstx, qq, f)
        filter_x.append(mx[0][0, 0])
        filter_y.append(mx[0][0, 1])
        filter_z.append(mx[0][0, 2])
        lstx = t
    gt_fig = plt.figure()
    ax = gt_fig.add_subplot(111, projection='3d')
    ax.plot(filter_x, filter_y, filter_z)
    ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('Filtered trajectory')
    ax.set_zlim(0, 5)
    plt.show()
