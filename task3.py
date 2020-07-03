import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random
import math
from scipy.linalg import norm
from scipy.interpolate import Rbf

def fun(t,x,A):
    return x.dot(A)


x0 = np.loadtxt(open("nonlinear_vectorfield_data_x0.txt", "rb"), delimiter=" ", skiprows=0)
x1 = np.loadtxt(open("nonlinear_vectorfield_data_x1.txt", "rb"), delimiter=" ", skiprows=0)


fig1, ax1 = plt.subplots()
ax1.scatter(x0[:,0],x0[:,1], s = 1,c = 'b')
ax1.scatter(x1[:,0],x1[:,1], s = 1, c= 'r')

dt = 0.1

F = (x1 - x0)/dt

fLin = np.linalg.lstsq(x0,F, rcond=None)
A = fLin[0]


fig2, ax2 = plt.subplots()


x1Aprox = []


for i in range(x0.shape[0]):
    sol = solve_ivp(fun = fun, t_span = [0,0.2], t_eval = [0.1], y0=x0[i, :], args = (A,))
    x1Aprox.append(sol.y)


x1Aprox = np.array(x1Aprox)
x1Aprox = x1Aprox.reshape((2000, 2))

mse = np.square(x1 - x1Aprox).mean()

ax2.scatter(x0[:,0],x0[:,1], s = 1, c = 'b')
ax2.scatter(x1Aprox[:,0],x1Aprox[:,1], s = 1, c = 'r')
ax2.scatter(x1[:,0],x1[:,1], s = 1, c = 'g')
print("Mean squared error for linear aprroximation = " +str(mse))



#eps = 0.01
#nCenters = 500
#centers = [np.random.uniform(-1, 1, 2000) for i in range(nCenters)]
##get phi values
#print(len(centers))
#Phi = np.empty((2000, nCenters))
#
#
#fLin = np.linalg.lstsq(phi,F, rcond=None)
#
#C_t = fLin[0]
#AA =  phi @ C_t
#
#
#
#x1AproxRBF = []
#fig3, ax3 = plt.subplots()
#for i in range(x0.shape[0]):
#    sol = solve_ivp(fun = fun, t_span = [0,10], t_eval = [10], y0=x0[i, :], args = (M,))
#    x1AproxRBF.append(sol.y)
#
#
#x1AproxRBF = np.array(x1AproxRBF)
#x1AproxRBF = x1AproxRBF.reshape((2000, 2))
#
#mse = np.square(x1 - x1AproxRBF).mean()
#
#ax3.scatter(x0[:,0],x0[:,1], s = 1, c = 'b')
#ax3.scatter(x1[:,0],x1[:,1], s = 1, c = 'g')
#ax3.scatter(x1AproxRBF[:,0],x1AproxRBF[:,1], s = 1, c = 'r')
#print("Mean squared error for RBF aprroximation = " +str(mse))
#
#fig4, ax4 = plt.subplots()
#y1 = np.arange(-1,1,0.01)
#y2 = np.arange(-1,1,0.01)
#
#Y1, Y2 = np.meshgrid(y1, y2)
#
#y1_dot = AA[0][0]*Y1 + AA[0][1]*Y2
#y2_dot = AA[1][0]*Y1 + AA[1][1]*Y2
#ax4.streamplot(Y1,Y2,y1_dot,y2_dot, density = 1,linewidth= 0.5)



plt.show()
