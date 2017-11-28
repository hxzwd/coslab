import cmath
import numpy as np
from lab1 import FT, FT1

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
