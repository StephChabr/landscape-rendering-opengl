import numpy as np

import matplotlib.pyplot as plt

cameras = np.load('E:\Stephane\data\set6\\final_cams.npy')

axX = cameras[:,1].reshape(-1)
axY = cameras[:,0].reshape(-1)
axT = cameras[:,3].reshape(-1)

for i in range(cameras.shape[0]):
    plt.arrow(axX[i], axY[i], 200*np.cos(axT[i]*np.pi/180), 200*np.sin(axT[i]*np.pi/180), width =30)
  

# Showing the graph
plt.show()
