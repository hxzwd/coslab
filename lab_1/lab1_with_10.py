import math
import cmath
import numpy as np
import os
import sys

from matplotlib import pyplot as plt


pi = math.pi
e = math.e

#тестовый сигнал, две гармоники и нормальный шум, чтобы не так скучно было
def sig(t):
	return 10*np.sin(2*pi*5*t) + 5*np.sin(2*pi*10*t) + np.random.normal(0.0, 1.0, (1, len(t))["array" in str(type(t_a)) or "list" in str(type(t_a))])
	

def sig2(t):
	return 10*np.sin(2*pi*5*t) + 5*np.sin(2*pi*10*t)
	
#начальное время наблюдения сигнала
t_0 = 0.0
#конечное время наблюдения сигнала
t_f = 5.0
#период дискретизации
T = 0.01

t_a = np.arange(t_0, t_f, T)
N = len(t_a)

#прямое ДПФ
#на вход отсчеты сигнала
def FT(x):
	return [ sum([ x[n]*cmath.exp(-1j*(2*pi/len(x))*k*n) for n in range(0, len(x)) ]) for k in range(0, len(x)) ]
#обратное ДПФ
#на вход отсчеты изображения Фурье сигнала	
def FT1(x):
	return list(map(lambda t: t.conjugate()/len(x), FT(list(map(lambda t: t.conjugate(), x)))))
#вычислить дискретный спектр, ну тут надо нормировку делать, я так в уире делал вроде
#вроде все правильно, но объяснить не могу почему так
#вернет амплитудный спектр, фазовый спектр и частоты в герцах для оси x спектра
#на вход отсчеты сигнала
def Sd(x):
	X = FT(x)
	Ph = [ cmath.phase(z) for z in X ]
	A = [ abs(z) for z in X ]
	A = A[ 0 : round(len(A)/2) ]
	Ph = Ph[ 0 : round(len(Ph)/2) ]
	A = [ a/len(A) for a in A ]
	F = [ f/(T*len(x)) for f in np.arange(0, len(A), 1.0) ]
	return [A, Ph, F]

#посчитать спектр непрерывного сигнала
#ну тут надо численно проинтегрировать методом трапеций
#вроде результат сходится с дискретным спектром
#на вход: сигнал, начальное время наблюдения, конечное время наблюдения
#шаг сетки по времени для интрегрирования
#частоты для которых считем спектр
def Sc(sg, t1, t2, ts, f):
	t = np.arange(t1, t2, ts)
	N = len(t)
	w = [ 2*pi*fr for fr in f ]
	tmp = []
	for w0 in w:
		x = sg(t)*np.exp(-1j*w0*t)
		tmp.append(sum( [ (x[i] + x[i + 1])*0.5*(t[i + 1] - t[i]) for i in range(0, N - 1) ] ))
	return [ [ abs(z)/(ts*round(N/2)) for z in tmp ], [ cmath.phase(z) for z in tmp ] ]

def Sinc(x):
	epsilon = 1e-6
	if abs(x) < epsilon:
		return 1
	else:
		return math.sin(x)/x
#восстанавливаем сигнал по теореме котельникова
#x - отсчёты
#t1, t2 - начальный и конечнй моменты времени
#ts - шаг по времени
#TT - период дискретизации
def TK(x, t1, t2, ts, TT):
	tmp = []
	qwe = []
	t = np.arange(t1, t2, ts)
	NN = len(t)
	K = len(x)
	for ind, t0 in enumerate(list(t)):
		qwe = []
		qwe = [ x[k]*Sinc(pi/TT*(float(t0) - float(k)*TT)) for k in range(0, K) ]
		tmp.append(sum(qwe))
	return tmp

#получить дискретный спектр из непрерывного
#принимает:
#сам сигнал
#начальное, конечное время наблюдения и шаг дескритизации сигнала
#начальную и конечную частоты спектра
#период дескритизации дискретного спектра
#возвращает амлитудный и фазовый дискретный спектр сигнала
def SdFromSc(sg, t1, t2, ts, f1, f2, df):
	Ad = []
	Pd = []
	f = np.arange(f1, f2, df)
	tmp = Sc(sg, t1, t2, ts, f)
	return tmp

x = sig(t_a)
[a, ph, f] = Sd(x)
#plt.figure(1)
#plt.plot(f, a)
#plt.show()

fc = np.arange(0.0, 50.0, 0.1)
[ac, phc] = Sc(sig, 0.0, 5.0, 0.01, fc)
#plt.figure(2)
#plt.plot(fc, ac)
#plt.show()

tt = np.arange(0.0, 1.0, 0.001)
xx = sig2(tt)
tt2 = np.arange(0.0, 1.0, 0.01)
xx2 = sig2(tt2)
xtk = TK(xx2, 0.0, 1.0, 0.001, 0.01)
#plt.figure(3)
#plt.plot(tt, xtk)
#plt.figure(4)
#plt.plot(tt, xx)

tmp = SdFromSc(sig2, 0.0, 1.0, 0.01, 0, 50.0, 1.0)
tmp1 = Sd(xx2)
#plt.figure(5)
#plt.plot(np.arange(0.0, 50.0, 1.0), tmp[0])
#plt.figure(6)
#plt.plot(tmp1[-1], tmp1[0])




#Восстановить сигнал по дискретному спектру
#A, Ph - амплитудный и фазовый дискретные спектры
#f - частоты спектра
#t0, t1, ts - начальное, конечное время и шаг по времени
def SdToSig(A, Ph, f = [], t0 = 0, t1 = 0, ts = 0):
	Tmp = [ cmath.rect(v*len(A), Ph[i]) for i, v in enumerate(A) ]	
	Tmp2 = [ cmath.rect(v*len(A), Ph[i]) for i, v in enumerate(A) ]
	Tmp2.reverse()
	Q = Tmp + np.conj(Tmp2).tolist()
	Q.pop()
	res = FT1(Q)
	return res


t_a = np.arange(0.01, 5, 0.01)
x = sig(t_a)
X = Sd(x)
x_from_Sd = np.real(SdToSig(X[0], X[1]))
#plt.figure(6)
#plt.plot(t_a, x)
#plt.figure(7)
#plt.plot(t_a, x_from_Sd)
