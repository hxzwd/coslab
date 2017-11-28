import numpy as np
from math import *
import cmath
import matplotlib.pyplot as plt
import scipy as sc

InfoMessage = "lab1_10_task.py"

def Sinc(x):
	return np.sinc(x/pi)


def Rect(x):
	epsilon = 10e-12
	if abs(x) > 0.5:
		return 0.0
	elif abs(abs(x) - 0.5) <= epsilon:
		return 0.5
	elif abs(x) < 0.5:
		return 1.0

def IntPart(x):
	return int(modf(x)[-1])

def Doc(x):
	print(x.__doc__)


print(InfoMessage)

a = pi/2.0	
x_func = lambda t: a/pi*Sinc(a*t)
S_func = lambda w: Rect(w/(2.0*a))

wb = -pi
we = pi
dw = 2*pi/1000
wa = np.arange(wb, we, dw)

tb = 0.0
te = 5.0
dt = 0.01
ta = np.arange(tb, te, dt)


S_d = np.array([ S_func(w) for w in wa ])
x_d = np.array([ x_func(t) for t in ta ])
N = len(S_d)

S_dh = S_d[round(N/2) : ]
W_dh = wa[round(N/2) : ]


x_0 = []
for t in ta:
	x_0.append(np.sum([ Sn*np.cos(W_dh[n]*t)/(len(S_dh)) for n, Sn in enumerate(S_dh) ]))

x_0 = np.array(x_0)


plt.figure(1)
plt.plot(W_dh, S_dh)

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(ta, x_d)
plt.subplot(2, 1, 2)
plt.plot(ta, x_0)

plt.figure(3)
p1 = plt.plot(ta, x_0, color = "red", linewidth = 4.0, label = "Rebuilt signal")
p2 = plt.plot(ta, x_d, color = "black", linewidth = 2.0, label = "True signal")
hndl = plt.gcf()
lg = hndl.legend(loc = "upper center", fontsize = "x-large")
lg.get_frame().set_facecolor('#00FFCC')

