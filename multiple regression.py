import numpy as nmp
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
 
def genrt_dataset(n):
    x = []
    y = []
    random_a1 = nmp.random.rand()
    random_a2 = nmp.random.rand()
    for i in range(n):
        a1 = i
        a2 = i/2 + nmp.random.rand()*n
        x.append([1, a1, a2])
        y.append(random_a1 * a1 + random_a2 * a2 + 1)
    return nmp.array(a), nmp.array(b)
 
a, b = genrt_dataset(200)
 
mpl.rcParams['legend.fontsize'] = 12
 
fig = plt.figure()
ax = fig.add_subplot(projection ='3d')
 
ax.scatter(a[:, 1], a[:, 2], b, label ='b', s = 5)
ax.legend()
ax.view_init(45, 0)
 
plt.show()