import numpy
from matplotlib import pyplot, cm
from numpy import linalg as LA
# Grid Definition
nx = 101
ny = 101
dx = 1. / (nx - 1)
dy = 1. / (ny - 1)
x = numpy.linspace(0, 1, nx)
y = numpy.linspace(0, 1, ny)
X, Y = numpy.meshgrid(x, x)

#define variables array
T=numpy.zeros((ny, nx))
Tt=numpy.zeros((ny, nx))
Ti=numpy.zeros((ny, nx))
#variables
Niter=50000
dt= 1e-5
a=1
rss=10
it=0

#Tc=numpy.random.randint(100, size=(100, 4))
Tc=numpy.zeros((100,4))
I=[15,2,5,10,20,50,75,100]

for k in range(1):
    
    rss=10
    T=numpy.zeros((ny, nx))
    Tt=numpy.zeros((ny, nx))
    Ti=numpy.zeros((ny, nx))
    
    '''#T-in
    Ti[0,:]    = Tc[k,0]
    Ti[nx-1,:] = Tc[k,1]
    Ti[:,0]    = Tc[k,2]
    Ti[:,ny-1] = Tc[k,3]'''
    

    # T
    T[0,:]    = 4
    T[nx-1,:] = 1
    T[:,0]    = 2
    T[:,ny-1] = 3
             
    while (rss>=1e-5):

        Tt=T.copy()      
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                T1=(Tt[i+1,j]-2*Tt[i,j]+Tt[i-1,j])/dx**2
                T2=(Tt[i,j+1]-2*Tt[i,j]+Tt[i,j-1])/dy**2
                T[i,j]=Tt[i,j]+dt*(T1+T2)
                 
        rss=   LA.norm(T-Tt) / LA.norm(Tt)
        print "Iter= %d\t rss=%.6f\t k=%d\t \n" %(it,rss,k)
        it=it+1
    #plot
    def plot(pp,name):
        pyplot.figure(figsize=(6, 5), dpi=100)
        cp = pyplot.tricontourf(X.flatten(),Y.flatten(),T.flatten())
        #pyplot.clabel(cp, inline=True,fontsize=8)
        pyplot.colorbar()
        pyplot.title('Contour Plot')
        pyplot.xlabel('X ')
        pyplot.ylabel('Y ')
        pyplot.savefig(name, format='png', dpi=100)
        pyplot.show()
   
    plot(T,'30t')
    
    fp=open('./data_file/laplace_internal.dat','w')
    for k in range(len(X.flatten())):
        fp.write('%f %f %f \n'%(X.flatten()[k],Y.flatten()[k],T.flatten()[k]))
    fp.close()
    
    fp=open('./data_file/laplace_boundary.dat','w')
    for k in range(len(T[0,:])):
        fp.write('%f %f %f \n'%(X[0,k],Y[0,k],T[0,k]))
    for k in range(len(T[0,:])):
        fp.write('%f %f %f \n'%(X[nx-1,k],Y[nx-1,k],T[nx-1,k]))        
    for k in range(len(T[0,:])):
        fp.write('%f %f %f \n'%(X[k,0],Y[k,0],T[k,0]))        
    for k in range(len(T[0,:])):
        fp.write('%f %f %f \n'%(X[k,ny-1],Y[k,ny-1],T[k,ny-1])) 
    fp.close()