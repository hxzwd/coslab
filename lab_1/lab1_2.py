import numpy as np
from math import *
import cmath
import matplotlib.pyplot as plt
import scipy as sc

InfoMessage = "lab1_2.py"

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

def z_ind(X, k, ind = False):
	M = len(X)
	if M % 2 == 1:
		ZI = list(range(-IntPart(M/2), IntPart(M/2) + 1))
	else:
		ZI = list(range(-IntPart(M/2), IntPart(M/2)))
	if ind:
		return ZI
	X1 = X[ZI.index(0) : ]
	X2 = X[: ZI.index(0) ]
	if k < 0:
		return X2[k]
	else:
		return X1[k]

def shuffle(X):
	Xc = np.copy(X).tolist()
	N = len(X)
	Tmp = np.random.random_integers(1, len(X)*3, size = len(X)*3)
	Tmp = [ z % len(X) for z in Tmp ]
	Tmp1 = []
	for j in Tmp:
		if j not in Tmp1:
			Tmp1.append(j)
	if len(Tmp1) < N:
		for j in range(0, N):
			if j not in Tmp1:
				Tmp1.append(j)
	Tmp = Tmp1
	for i in range(0, N):
		tmp = Xc[i]
		Xc[i] = Xc[Tmp[i]]
		Xc[Tmp[i]] = tmp
	return (Tmp, Xc, X)	

def DFT(x, Indexes):
	X = []
	N = len(Indexes)
	for n in Indexes:
		X.append(np.sum([ z_ind(x, k)*cmath.exp(-1j*2*pi*k*n/N) for k in Indexes ]))	
	return (X, Indexes)	

def test():
#	test_x_func = lambda t: \
#	10*sin(2*pi*5*t + pi/2) + 5*sin(2*pi*10*t + 3*pi/4)
#	tb = 0.0
#	te = 10.0
#	dt = 0.01
#	ta = np.arange(tb, te, dt)
#	x = [ test_x_func(t) for t in ta ]
#	X = np.fft.fft(x)
#	A0 = np.abs(X)
#	Phi0 = np.array([ cmath.phase(z) for z in X ])
#	A = A0[ : round(len(A0)/2.0) ]
#	A = A/len(A)
#	Phi = Phi0[ : len(A) ]
#	Freqs = np.array([ freq/(dt*len(X)) for freq in np.arange(0.0, len(A), 1.0) ])
#	W = 2*pi*Freqs
#	return {"A":A, "P":Phi, "F":Freqs, "W":W, "T":ta}
	a = pi/2.0	
	test_x_func = lambda t: a/pi*Sinc(a*t)
	test_S_func = lambda w: Rect(w/(2.0*a))
	wb = -pi
	we = pi
	dw = 2*pi/1000
	wa = np.arange(wb, we, dw)
	tb = -10.0
	te = 10.0
	dt = 0.01
#	tb = 0.0
	ta = np.arange(tb, te, dt)
	test_S = np.array([ test_S_func(w) for w in wa ])
	test_x = np.array([ test_x_func(t) for t in ta ])
#	Indexes = z_ind(test_x, 0, ind = True)
	Indexes = []
	test_x0 = []
#	for t in ta:
#		Tmp = np.array([ v*cmath.exp(1j*v*t*wa[i]) for i, v in enumerate(test_S) ])
#		test_x0.append(1/(2*pi)*np.trapz(Tmp, wa))
	Tmp0 = []
	tau = te
	test_x1 = []
	for t in ta:
		Tmp0 = (4*2*pi)*np.array([ test_S[i]*cmath.exp(-1j*v*t)*Rect(t*tau/2) for i, v in enumerate(wa)  ])
		test_x1.append(np.sum(Tmp0))
	return { "S":test_S, "W":wa, "T":ta, "x":test_x, "x_ind":Indexes, "x0":test_x0, "x1":test_x1 }



def S_func(w):
	return cmath.exp(-1j*w)

def x_func(t):
	epsilon = 1e-6
	if abs(t - 1.0) <= epsilon:
		return sqrt(pi/2.0)
	else:
		return sqrt(2.0/pi)*sin(pi/2.0*(t - 1.0))/(t - 1.0)


print(InfoMessage)

wb = -pi/2
we = pi/2
dw = pi/500

ta = np.arange(0.0, 5.0, 0.01)
wa = np.arange(wb, we, dw)
ta1 = np.arange(-5.0, 5.0, 0.01)

x = [ x_func(t) for t in ta ]
x1 = [ x_func(t) for t in ta1 ]
S = [ S_func(w) for w in wa ]

A = np.real(S)
Phi = np.imag(S)

N = len(A)

xf_r = lambda t: sum([ A[n]*cos(wa[n]*t + Phi[n]) for n in range(0, N) ])

