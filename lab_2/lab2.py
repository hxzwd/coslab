import matplotlib.pyplot as plt
import numpy as np
from math import *

L_begin = 1
L_end = 1000
La = range(L_begin, L_end)

N_val = 1000
M_val = 4
L_val_begin = N_val
L_val_end = 5000
N_val_begin = M_val
N_val_end = N_val

f1 = lambda L: L**2 + (L - 1)**2
f2 = lambda L: 3*L*(6*log2(L) - 7) + 24
f3 = lambda L: 12*L*log2(L) - 22.25*L + 28

g1 = lambda N, M, L: 2*L*M + M - L - (M - 1)**2
g2 = lambda N, M, n: 2*n*(N*M - M + 1) - (N - M + 1)
g3 = lambda N, M, n: 2*n*N*(N + M) - 3*n*N
L_g2 = lambda N, M, n: n*(N + M - 1) - (N - M + 1)
L_g3 = lambda N, M, n: n*N
L_g2_1 = lambda N, M, L: float((L + (N - M + 1)))/float((N + M - 1))
L_g3_1 = lambda N, M, L: float(L)/float(N)

def lab3(Lb = L_begin, Le = L_end):
	La_ = range(Lb, Le)
	P1 = [ f1(L) for L in La_ ]
	P2 = [ f2(L) for L in La_ ]
	P3 = [ f3(L) for L in La_ ]
	plt.grid(True)
	p1, = plt.plot(La_, P1, label = 'Convolution by definition')
	p2, = plt.plot(La_, P2, label = 'Convolution by FFT')
	p3, = plt.plot(La_, P3, label = 'Convolution by FHT')
	plt.legend([p1, p2, p3], ['Convolution by definition', 'Convolution by FFT', 'Convolution by FHT'])
	return [P1, P2, P3], plt.gcf()
	
def lab2(Nvb = N_val_begin, Nve = N_val_end, Mv = M_val, Lvb = L_val_begin, Lve = L_val_end):
	L_ = Lve
	M_ = Mv
	Na = range(Nvb, Nve)
	P1 = [ g1(N, M_, L_) for N in Na ]
	P2 = [ g2(N, M_, round(L_g2_1(N, M_, L_))) for N in Na ]
	P3 = [ g3(N, M_, round(L_g3_1(N, M_, L_))) for N in Na ]
	plt.grid(True)
	p1, = plt.plot(Na, P1, label = 'Convolution by definition')
	p2, = plt.plot(Na, P2, label = 'Overlap with summation')
	p3, = plt.plot(Na, P3, label = 'Overlap with accumulation')
	plt.legend([p1, p2, p3], ['Convolution by definition', 'Overlap with summation', 'Overlap with accumulation'])
	return [P1, P2, P3, Na], plt.gcf()

def saveplot(Fig, filename = '', Dpi = 100):
	if filename == '':
		filename = 'fig_' + str(Fig.number) + '.png'
	Fig.savefig(filename, dpi = Dpi)