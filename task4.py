import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import math



def lorenz(x0, mTime, sigma, beta, rho):

    def lorenz_deriv(t0, x_y_z, sigma=sigma, beta=beta, rho=rho):
        """Compute the time-derivative of a Lorenz system."""
        x, y, z = x_y_z
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    t = np.linspace(0, mTime, 1500)
    x_t = np.asarray([solve_ivp(fun=lorenz_deriv, t_span=[0, 1500], y0=val, t_eval=t) for val in x0])


    return t, x_t





data = np.loadtxt(open("takens_1.txt", "rb"), delimiter=" ", skiprows=0)

x = data[:,0]
y = data[:,1]

fig1, ax1 = plt.subplots()

ax1.plot(range(1000),x)
ax1.set_title(' plot of the first coordinate against the line number in the dataset')
ax1.set_xlabel('time')
ax1.set_ylabel('first coordinate')

fig2, ax2 = plt.subplots()
delay = 50
points = 334
ax2.plot(x[0:points],x[delay:delay+points])
ax2.set_xlabel('first coordinate')
ax2.set_title(str(points) + ' Points needed with delay = ' + str(delay))
ax2.set_ylabel('delayed first coordinate')

fig3, ax3 = plt.subplots()
ax3.plot(x,y)
ax3.set_xlabel('first coordinate')
ax3.set_ylabel('second coordinate')

fig4 = plt.figure()
ax4 = fig4.gca(projection='3d')

x0 = [[10,10,10]]
sigma = 10
beta = 8.0/3.0
rho  = 28
t, sol= lorenz(x0, mTime=30,sigma = sigma,rho = rho, beta = beta)


for i in range(len(x0)):
    x = sol[i].y[0, :]
    y = sol[i].y[1, :]
    z = sol[i].y[2, :]
    lines = ax4.plot(x, y, z)
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_zlabel("z")


delay = 3
size = 500

#delayed x
x1 = x[:size]
x2 = x[delay:size+delay]
x3 = x[2*delay:size+2*delay]


fig5, ax5 = plt.subplots(3)
ax5[0].plot(range(size),x1)
ax5[1].plot(range(size),x2)
ax5[2].plot(range(size),x3)
ax5[0].set_title(r"Delayed graphs $x_1,x_2,x_3$")
ax5[0].set_xlabel('time')
ax5[1].set_xlabel('time')
ax5[2].set_xlabel('time')
ax5[0].set_ylabel(r'$x_1$')
ax5[1].set_ylabel(r'$x_2$')
ax5[2].set_ylabel(r'$x_3$')


fig6 = plt.figure()
ax6 = fig6.gca(projection='3d')
lines = ax6.plot(x1, x2, x3)
ax6.set_xlabel(r'$x_1$')
ax6.set_ylabel(r'$x_2$')
ax6.set_zlabel(r'$x_3$')

z1 = z[:size]
z2 = z[delay:size+delay]
z3 = z[2*delay:size+2*delay]

fig7 = plt.figure()
ax7 = fig7.gca(projection='3d')
lines = ax7.plot(z1, z2, z3)
ax7.set_xlabel(r'$z_1$')
ax7.set_ylabel(r'$z_2$')
ax7.set_zlabel(r'$z_3$')






plt.show()
#plt.setp(lines, linewidth=2)
