import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd
import math



data = np.loadtxt(open("MI_timesteps.txt", "rb"), delimiter=" ", skiprows=1)


time = data[:,0]
col2 = data[:,1]
col3 = data[:,2]
col4 = data[:,3]
col5 = data[:,4]
col6 = data[:,5]
col7 = data[:,6]
col8 = data[:,7]
col9 = data[:,8]
col10 = data[:,9]


fig1, ax1 = plt.subplots(9)

ax1[0].plot(time, col2,c = 'r')
ax1[1].plot(time, col3, c='b')
ax1[2].plot(time, col4, c = 'g')
ax1[3].plot(time, col5, c = 'navy')
ax1[4].plot(time, col6, c = 'orange')
ax1[5].plot(time, col7, c = 'purple')
ax1[6].plot(time, col8, c = 'salmon')
ax1[7].plot(time, col9, c = 'gold')
ax1[8].plot(time, col10, c = 'black')
ax1[0].set_ylabel('Col 2')
ax1[1].set_ylabel('Col 3')
ax1[2].set_ylabel('Col 4')
ax1[3].set_ylabel('Col 5')
ax1[4].set_ylabel('Col 6')
ax1[5].set_ylabel('Col 7')
ax1[6].set_ylabel('Col 8')
ax1[7].set_ylabel('Col 9')
ax1[8].set_ylabel('Col 10')
ax1[8].set_xlabel('time')

windows = []
for i in range(1000,len(col2)):
    window = [col2[i:i + 351], col3[i:i + 351], col4[i:i + 351]]
    windows.append(np.array(window).T.flatten())
    if i + 351 >= len(col2):
        break
windows = np.array(windows)

centered_data = (windows - windows.mean())
pca = PCA(n_components=3)
#pc = pca.fit_transform(centered_dataset - centered_dataset.mean())

PCs = pca.fit_transform(windows)
print(len(windows))

ax2 = plt.figure().gca(projection='3d')
#ax2.plot(PCs[:,0],PCs[:,1],PCs[:,2])
ax2.scatter(PCs[:, 0],PCs[:, 1],PCs[:, 2], s= 1)
ax2.set_xlabel('principal component 1')
ax2.set_ylabel('principal component 2')
ax2.set_zlabel('principal component 3')

fig = plt.figure()

ax3 =  fig.add_subplot(331, projection='3d')
ax3.scatter(PCs[:, 0],PCs[:, 1],PCs[:, 2], s= 1,  cmap='jet' ,c=data[:len(windows), 1])
ax3.set_title('Col2')

ax4 = fig.add_subplot(332, projection='3d')
ax4.scatter(PCs[:, 0],PCs[:, 1],PCs[:, 2], s= 1,  cmap='jet' ,c=data[:len(windows), 2])
ax4.set_title('Col3')
ax5 = fig.add_subplot(333, projection='3d')
ax5.scatter(PCs[:, 0],PCs[:, 1],PCs[:, 2], s= 1,  cmap='jet' ,c=data[:len(windows), 3])
ax5.set_title('Col4')

ax6 = fig.add_subplot(334, projection='3d')
ax6.scatter(PCs[:, 0],PCs[:, 1],PCs[:, 2], s= 1,  cmap='jet' ,c=data[:len(windows), 4])
ax6.set_title('Col5')

ax7 = fig.add_subplot(335, projection='3d')
ax7.scatter(PCs[:, 0],PCs[:, 1],PCs[:, 2], s= 1,  cmap='jet' ,c=data[:len(windows), 5])
ax7.set_title('Col6')

ax8 = fig.add_subplot(336, projection='3d')
ax8.scatter(PCs[:, 0],PCs[:, 1],PCs[:, 2], s= 1,  cmap='jet' ,c=data[:len(windows), 6])
ax8.set_title('Col7')

ax9 = fig.add_subplot(337, projection='3d')
ax9.scatter(PCs[:, 0],PCs[:, 1],PCs[:, 2], s= 1,  cmap='jet' ,c=data[:len(windows), 7])
ax9.set_title('Col8')

ax10 = fig.add_subplot(338, projection='3d')
ax10.scatter(PCs[:, 0],PCs[:, 1],PCs[:, 2], s= 1,  cmap='jet' ,c=data[:len(windows), 8])
ax10.set_title('Col9')

ax11 = fig.add_subplot(339, projection='3d')
ax11.scatter(PCs[:, 0],PCs[:, 1],PCs[:, 2], s= 1,  cmap='jet' ,c=data[:len(windows), 9])
ax11.set_title('Col10')

#arc arc_length


points = 1990
arcLength = 0
for i in range(points):
    arcLength += math.sqrt(PCs[i][0]**2 + PCs[i][1]**2 + PCs[i][2]**2)
ax12 = plt.figure().gca(projection='3d')
ax12.plot(PCs[:points, 0],PCs[:points, 1],PCs[:points, 2])
ax12.set_title('Arc length = '+str(arcLength))


#ax4.streamplot(Y1,Y2,y1_dot,y2_dot, density = 1,linewidth= 0.5)


plt.show()
