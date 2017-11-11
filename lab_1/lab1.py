import math
import cmath
import numpy as np
import os
import sys

from matplotlib import pyplot as plt

pi = math.pi
e = math.e

#�������� ������, ��� ��������� � ���������� ���, ����� �� ��� ������ ����
def sig(t):
	return 10*np.sin(2*pi*5*t) + 5*np.sin(2*pi*10*t) + np.random.normal(0.0, 1.0, (1, len(t))["array" in str(type(t_a)) or "list" in str(type(t_a))])
	
#��������� ����� ���������� �������
t_0 = 0.0
#�������� ����� ���������� �������
t_f = 5.0
#������ �������������
T = 0.01

t_a = np.arange(t_0, t_f, T)
N = len(t_a)

#������ ���
#�� ���� ������� �������
def FT(x):
	return [ sum([ x[n]*cmath.exp(-1j*(2*pi/len(x))*k*n) for n in range(0, len(x)) ]) for k in range(0, len(x)) ]
#�������� ���
#�� ���� ������� ����������� ����� �������	
def FT1(x):
	return list(map(lambda t: t.conjugate()/len(x), FT(list(map(lambda t: t.conjugate(), x)))))
#��������� ���������� ������, �� ��� ���� ���������� ������, � ��� � ���� ����� �����
#����� ��� ���������, �� ��������� �� ���� ������ ���
#������ ����������� ������, ������� ������ � ������� � ������ ��� ��� x �������
#�� ���� ������� �������
def Sd(x):
	X = FT(x)
	Ph = [ cmath.phase(z) for z in X ]
	A = [ abs(z) for z in X ]
	A = A[ 0 : round(len(A)/2) ]
	Ph = Ph[ 0 : round(len(Ph)/2) ]
	A = [ a/len(A) for a in A ]
	F = [ f/(T*len(x)) for f in np.arange(0, len(A), 1.0) ]
	return [A, Ph, F]

#��������� ������ ������������ �������
#�� ��� ���� �������� ���������������� ������� ��������
#����� ��������� �������� � ���������� ��������
#�� ����: ������, ��������� ����� ����������, �������� ����� ����������
#��� ����� �� ������� ��� ���������������
#������� ��� ������� ������ ������
def Sc(sg, t1, t2, ts, f):
	t = np.arange(t1, t2, ts)
	N = len(t)
	w = [ 2*pi*fr for fr in f ]
	tmp = []
	for w0 in w:
		x = sg(t)*np.exp(-1j*w0*t)
		tmp.append(sum( [ (x[i] + x[i + 1])*0.5*(t[i + 1] - t[i]) for i in range(0, N - 1) ] ))
	return [ [ abs(z)/(ts*round(N/2)) for z in tmp ], [ cmath.phase(z) for z in tmp ] ]

#��-�� �� �������� ����
#������� ������������ ������ �� �������
#�� ������� �� ������� ������������
#������ �� ���� ������� ���-�� ��������
qwe = []
def TK(x, t1, t2, ts):
	tmp = []
	t = np.arange(t1, t2, ts)
	N = len(t)
	K = len(x)
	for ind, t0 in enumerate(list(t)):
		qwe = [ x[k]*(math.sin((pi/T)*(t0 - k*T))/((pi/T)*(t0 - k*T)), 1)[abs(t0 - k*T) < 0.00001] for k in range(0, K) ]
		tmp.append(sum(qwe))
	return tmp
	
x = sig(t_a)
[a, ph, f] = Sd(x)
plt.plot(f, a)
plt.show()

fc = np.arange(0.0, 50.0, 0.1)
[ac, phc] = Sc(sig, 0.0, 5.0, 0.01, fc)
plt.plot(fc, ac)
plt.show()

#xtk = TK(x, 0.0, 1.0, 0.01)