import math
import numpy as np
import matplotlib.pyplot as plt

# Landmarks
marks = []
with open("lab1\\data files\\landmarks.dat") as file:
    for line in file.readlines():
        coord = line.split()
        marks.append([float(coord[1]), float(coord[2])])

# Params
Q = np.array([[0.2, 0, 0],
                [0, 0.2, 0],
                [0, 0, 0.2]])

def upd(state, u, ind = 0):
    state[ind, 0] += u[1] * math.cos(state[ind, 2] + u[0])
    state[ind, 1] += u[1] * math.sin(state[ind, 2] + u[0])
    state[ind, 2] += u[1] + u[2]
    return state

def Fx(state, u):
    return np.array([[1.0, 0, -u[1] * math.sin(state[0, 2] + u[0])],
                      [0, 1, u[1] * math.cos(state[0, 2] + u[0])],
                      [0, 0, 1]])

def kalman(obs, m, P, u):
    R = np.eye(len(obs)) * 0.2
    # Prediction
    F = Fx(m, u)
    m = upd(m, u)
    P = F @ P @ F.T + Q
    # h and H calculating
    h = np.zeros((1, len(obs)))
    H = np.zeros((len(obs), 3))
    y = np.zeros((1, len(obs)))
    nxt = 0
    for i, r in obs:
        d_x = m[0, 0] - marks[i - 1][0]
        d_y = m[0, 1] - marks[i - 1][1]
        h[0, nxt] = math.sqrt(d_x ** 2 + d_y ** 2)
        H[nxt, :] = [d_x / h[0, nxt], d_y / h[0, nxt], 0]
        y[0, nxt] = r
        nxt += 1
    # Update
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    m = m + (K @ (y - h).T).T
    P = P - K @ S @ K.T
    return m, P

def ukalman(obs, m, P, u):
    # Prediction
    n = 3
    R = np.eye(len(obs)) * 0.2
    lam = 1
    sig = np.zeros((2 * n + 1, n))
    sig[0, :] = np.array(m)
    W = [1 / 2 / (n + lam) for _ in range(2 * n + 1)]
    W[0] = lam / (n + lam)
    C = np.linalg.cholesky(P)
    for i in range(n):
        sig[i + 1, :] = m + (math.sqrt(n + lam) * C[:, i]).T
        sig[i + n + 1, :] = m - (math.sqrt(n + lam) * C[:, i]).T
    for i in range(2 * n + 1):
        sig = upd(sig, u, i)
    m = np.zeros((1, n))
    P = np.array(Q)
    for i in range(2 * n + 1):
        m = m + sig[i, :] * W[i]
    for i in range(2 * n + 1):
        P = P + W[i] * (sig[i, :] - m).T @ (sig[i, :] - m)
    # Update
    sig = np.zeros((2 * n + 1, n))
    sig[0, :] = np.array(m)
    C = np.linalg.cholesky(P)
    for i in range(n):
        sig[i + 1, :] = m + (math.sqrt(n + lam) * C[:, i]).T
        sig[i + n + 1, :] = m - (math.sqrt(n + lam) * C[:, i]).T
    hsig = np.zeros((len(sig), len(obs)))
    y = np.zeros((1, len(obs)))
    for i in range(2 * n + 1):
        nxt = 0
        for j, r in obs:
            d_x = sig[i, 0] - marks[j - 1][0]
            d_y = sig[i, 1] - marks[j - 1][1]
            hsig[i, nxt] = math.sqrt(d_x ** 2 + d_y ** 2)
            y[0, nxt] = r
            nxt += 1
    mu = np.zeros((1, len(obs)))
    S = np.array(R)
    C = np.zeros((n, len(obs)))
    for i in range(2 * n + 1):
        mu = mu + W[i] * hsig[i, :]
    for i in range(2 * n + 1):
        S = S + W[i] * (hsig[i, :] - mu).T @ (hsig[i, :] - mu) 
        C = C + W[i] * (sig[i, :] - m).T @ (hsig[i, :] - mu)
    K = C @ np.linalg.inv(S)
    m = m + (K @ (y - mu).T).T
    P = P - K @ S @ K.T
    return m, P

# EKF
ekf_x, ekf_y = [], []
with open("D:\everything\code\lab\signals\lab1\data files\sensor_data_ekf.dat") as file:
    sensors = []
    u = np.array([])
    m = np.array([[0, 0, 0]])
    P = np.eye(3) * 0.01
    for line in file.readlines():
        query = line.split()
        if query[0] == "ODOMETRY":
            if len(u) > 0:
                m, P = kalman(sensors, m, P, u)
                ekf_x.append(m[0, 0])
                ekf_y.append(m[0, 1])
            sensors.clear()
            u = np.array([float(query[1]), float(query[2]), float(query[3])])
        elif query[0] == "SENSOR":
            sensors.append([int(query[1]), float(query[2])])
    m, P = kalman(sensors, m, P, u)
    ekf_x.append(m[0, 0])
    ekf_y.append(m[0, 1])

# UKF
ukf_x, ukf_y = [], []
with open("D:\everything\code\lab\signals\lab1\data files\sensor_data_ekf.dat") as file:
    sensors = []
    u = np.array([])
    m = np.array([[0, 0, 0]])
    P = np.eye(3) * 0.01
    for line in file.readlines():
        query = line.split()
        if query[0] == "ODOMETRY":
            if len(u) > 0:
                m, P = ukalman(sensors, m, P, u)
                ukf_x.append(m[0, 0])
                ukf_y.append(m[0, 1])
            sensors.clear()
            u = np.array([float(query[1]), float(query[2]), float(query[3])])
        elif query[0] == "SENSOR":
            sensors.append([int(query[1]), float(query[2])])
    m, P = ukalman(sensors, m, P, u)
    ukf_x.append(m[0, 0])
    ukf_y.append(m[0, 1])

fig, ax = plt.subplots(1, 2)
ax[0].plot(ekf_x, ekf_y)
ax[0].set_title("EKF")
ax[1].plot(ukf_x, ukf_y)
ax[1].set_title("UKF")
plt.show()