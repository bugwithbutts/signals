import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import copy

# Sensors
r = []
lattice = 5
for i in range(lattice):
    for j in range(lattice):
        r.append([-2 + i * 4 / lattice, -2 + j * 4 / lattice])

# Params
N = 50
L = len(r)
lam = 0.05
sig = 0.05
mu = 1000
C = np.array([[lam, 0, 0, 0],
               [0, lam, 0, 0],
               [0, 0, sig, 0],
               [0, 0, 0, sig]])
G = np.eye(L) * 0.001
h = 10
radius = 2

def upd(state, t):
    state[0, 0] = radius * math.cos(t * 2 * math.pi / 360)
    state[0, 1] = radius * math.sin(2 * t * math.pi / 360)
    state[0, 2] += np.random.normal(0, C[2, 2])
    state[0, 3] += np.random.normal(0, C[3, 3])
    # for i in range(4):
    #     state[0, i] += np.random.normal(0, C[i, i])
    return state

def observe(state):
    b = np.zeros((1, L))
    for i in range(L):
        d = np.sqrt((r[i][0] - state[0, 0]) ** 2 + (r[i][1] - state[0, 1]) ** 2 + h ** 2) ** 3
        b[0, i] = mu * ((r[i][1] - state[0, 1]) * state[0, 2] - (r[i][0] - state[0, 0]) * state[0, 3]) / d
        b[0, i] += np.random.normal(0, G[i, i])
    return b

def H(state):
    s1 = np.zeros((L))
    s2 = np.zeros((L))
    for i in range(L):
        d = np.sqrt((r[i][0] - state[0, 0]) ** 2 + (r[i][1] - state[0, 1]) ** 2 + h ** 2) ** 3
        s1[i] = mu * (r[i][0] - state[0, 0]) / d
        s2[i] = mu * (r[i][1] - state[0, 1]) / d
    return np.array([s2, -s1]).T

def kalman(m, P, u, w, y):
    A = np.eye(2)
    Q = np.eye(2) * C[3, 3]
    # Prediction
    m = (A @ m.T).T
    P = A @ P @ A.T + Q
    # Gen
    u[0, 0] += np.random.normal(0, C[0, 0])
    u[0, 1] += np.random.normal(0, C[1, 1])
    Hx = H(u)
    # Weight
    S = Hx @ P @ Hx.T + G
    w = multivariate_normal.pdf(y[0, :], mean=(Hx @ m.T).T[0], cov=S)
    # Update
    K = P @ Hx.T @ np.linalg.inv(S)
    m = m + (K @ (y.T - Hx @ m.T)).T
    P = P - K @ S @ K.T
    return [m, P, u, w]

def resample(parts, www):
    tmp = [(k + np.random.uniform(low=0.0, high=1.0)) / N for k in range(N)]
    tmp.sort()
    nw = []
    acc = 0
    cur = 0
    for m, P, u, w in parts:
        while cur < N and tmp[cur] > acc and tmp[cur] <= acc + w / www:
            nw.append(copy.deepcopy([m, P, u, w]))
            cur += 1
        acc += w / www
    return nw

# Part := m, P, u, w
parts = [[np.array([[1.0, 2]]), np.eye(2) * 0.01, np.array([[radius, 0.0]]), 1.0 / N] for _ in range(N)]
state = np.array([[radius, 0.0, 1, 2]])
state_x, state_y = [], []
filter_x, filter_y = [], []
for t in range(370):
    state = upd(state, t)
    y = observe(state)
    state_x.append(state[0, 0])
    state_y.append(state[0, 1])
    avr = np.zeros((1, 2))
    www = 0.0
    nw_parts = []
    for m, P, u, w in parts:
        m, P, u, w = kalman(m, P, u, w, y)
        avr += u * w
        www += w
        nw_parts.append(copy.deepcopy([m, P, u, w]))
    parts = nw_parts
    avr /= www
    parts = resample(parts, www)
    filter_x.append(avr[0, 0])
    filter_y.append(avr[0, 1])
    
fig, ax = plt.subplots(1, 2)
ax[0].plot(state_x, state_y)
ax[0].set_title("Path")
ax[1].plot(filter_x, filter_y)
ax[1].set_title("Filtered")
plt.show()
