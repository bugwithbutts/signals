import math
import numpy as np
import matplotlib.pyplot as plt

# Params
T = 1
st = 5
sr = 4
R = np.eye(3) * 0.5

def upd(state):
    state[0, 0] += T * st * math.cos(state[0, 2]) - 0.5 * T * T * st * sr * math.sin(state[0, 2])
    state[0, 1] += T * st * math.sin(state[0, 2]) + 0.5 * T * T * st * sr * math.cos(state[0, 2])
    state[0, 2] += T * sr
    return state

def observe(state):
    obs = np.array(state)
    obs[0, 0] += np.random.normal(0, R[0, 0], 1)[0]
    obs[0, 1] += np.random.normal(0, R[1, 1], 1)[0]
    obs[0, 2] += np.random.normal(0, R[2, 2], 1)[0]
    return obs

def Fx(state):
    return np.array([[1.0, 0, -T * st * math.sin(state[0, 2]) - 0.5 * T * T * st * sr * math.cos(state[0, 2])],
                      [0, 1, T * st * math.cos(state[0, 2]) - 0.5 * T * T * st * sr * math.sin(state[0, 2])],
                      [0, 0, 1]])

def Hx(state):
    return np.eye(3)

def kalman(state, m, P):
    # Prediction
    F = Fx(m)
    m = upd(m)
    H = Hx(m)
    P = F @ P @ F.T
    # Update
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    m = m + (K @ (state - m).T).T
    P = P - K @ S @ K.T
    return [m, P]

state = np.array([[0.0, 0.0, 0.0]])
state_x, state_y = [], [] 
filter_x, filter_y = [], []
obs_x, obs_y = [], []
m = np.zeros((1, 3))
P = np.eye(3) * 0.01

for _ in range(100):
    state = upd(state)
    state_x.append(state[0, 0])
    state_y.append(state[0, 1])

    obs = observe(state)
    obs_x.append(obs[0, 0])
    obs_y.append(obs[0, 1])

    m, P = kalman(obs, m, P)
    filter_x.append(m[0, 0])
    filter_y.append(m[0, 1])

fig, ax = plt.subplots(1, 3)
ax[0].plot(state_x, state_y)
ax[0].set_title("Path")
ax[1].plot(obs_x, obs_y)
ax[1].set_title("Observations")
ax[2].plot(filter_x, filter_y)
ax[2].set_title("Filtered")
plt.show()