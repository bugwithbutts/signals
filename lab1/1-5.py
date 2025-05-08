import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import re

# Params
V = 1125    # Sound speed
F = 100000  # Frequency
N = 500
speakers = np.array([[0, 0, 10], [20, 0, 10], [20, 20, 10], [0, 20, 10]])

f = open('lab1\\data files\\Transmitter.txt', 'r')
text = f.read()
f.close()
rows =  text.split('\n')

s_records = np.zeros((N, 4), dtype=np.float64)
for i in range(0, 4):
    str_array = np.array(re.split('[ \t]+', rows[i]))[1:]
    num_array = str_array.astype(np.float64)
    s_records[:, i] = num_array

f = open('lab1\\data files\\Receiver.txt', 'r')
text = f.read()
f.close()
received_txt = np.array(re.split('[ \t]+', text))[1:]
received = received_txt.astype(np.float64) 

dist = np.zeros(4) 
for i in range(0, 4):
    cor = sc.signal.correlate(received, s_records[:,i])
    lags = sc.signal.correlation_lags(len(received), len(s_records[:,i]))
    id = np.argmax(cor)
    T = lags[id] / F
    dist[i] = V * T

def minimize(x):
    dst_pos = np.zeros(4)
    for i in range(0, 4):
        dst_pos[i] = np.linalg.norm(speakers[i] - x)
    return dst_pos - dist

ret = sc.optimize.least_squares(minimize, [1, 1, 1])
print(ret.x)