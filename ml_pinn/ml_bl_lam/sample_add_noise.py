import numpy as np
from matplotlib import pyplot as plt


np.random.seed(137)
xy=np.loadtxt('./data_file/Re100_5p/bl_sample__no_error_x8.dat',skiprows=1)

fp=open('./data_file/Re100_5p/bl_sample_25p_noise_x8.dat','w')
fp.write('x y p u v @ x= 0,0.5,1,.,,,4.5, y: d/2... \n')

x = xy[:,0]
p = xy[:,2]
percentage = 0.25
n = np.random.normal(0, p.std(), x.size) * percentage
p3 = p + n

x = xy[:,0]
p = xy[:,3]
percentage = 0.25
n = np.random.normal(0, p.std(), x.size) * percentage
p4 = p + n

x = xy[:,0]
p = xy[:,4]
percentage = 0.25
n = np.random.normal(0, p.std(), x.size) * percentage
p5 = p + n

for i in range(len(xy)):
    fp.write('%f %f %f %f %f\n'%(xy[i,0],xy[i,1],p3[i],p4[i],p5[i]))
    
fp.close()    


plt.figure()
plt.tricontourf(xy[:,0],xy[:,1],xy[:,3])
plt.show()

plt.figure()
plt.tricontourf(xy[:,0],xy[:,1],p4)
plt.show()
