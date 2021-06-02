#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide="ignore",invalid="ignore")


# 
# # Boundary element method meshing

# In[2]:


def geometry(x_list,y_list,seg_list):
    Ns=np.sum(seg_list)  #total no. of segments
    Np=int(Ns+1) #total no. of segments including end points
    
    lb= np.sqrt((x_list[1:]-x_list[:-1])**2+(y_list[1:]-y_list[:-1])**2)
    #meshing
    seg_num=np.zeros(seg_list.size)
    for i in range(1,seg_list.size):
        seg_num[i]=seg_num[i-1]+seg_list[i-1]
    #creating empty arrays to store the x&y coordinates of nodes 
    x,y = [np.zeros(Np) for i in range(2)]
    #storing the initial value of the nodes
    x[0]=x[-1]=x_list[0]
    y[0]=y[-1]=y_list[0]
    
    #performing the BEM meshing 
    for i in range(seg_list.size):
        x[int(seg_num[i]):int(seg_num[i]+seg_list[i]+1)]=np.linspace((x_list[i]),(x_list[i+1]),(seg_list[i]+1))
        y[int(seg_num[i]):int(seg_num[i]+seg_list[i]+1)]=np.linspace((y_list[i]),(y_list[i+1]),(seg_list[i]+1))
    
    #midpoints for the elements
    xm=0.5*(x[1:]+x[:-1])
    ym=0.5*(y[1:]+y[:-1])
    
    xms,yms = [[0]*seg_list.size for i in range(2)]
    for i in range(seg_list.size):
        xms[i]=np.array(xm[int(seg_num[i]):int(seg_num[i]+seg_list[i])])
        yms[i]=np.array(ym[int(seg_num[i]):int(seg_num[i]+seg_list[i])])
        
    #length of the segment
    l=np.sqrt((x[1:]-x[:-1])**2+(y[1:]-y[:-1])**2)
    
    #normal vectors
    ny = (x[:-1]-x[1:])/l
    nx = (y[1:]-y[:-1])/l
    return x,y,xm,ym,xms,yms,nx,ny,l,Ns,seg_num,lb


#Boundary conditions for each segment or element
def SetBc(bct,bcv,seg_list,seg_num):
    #creating an empty array to store values
    BCT,BCV = [np.zeros(Ns) for i in range(2)]
    
    #setting the boundary conditions for each segment
    for i in range(seg_list.size):
        BCT[int(seg_num[i]):int(seg_num[i]+seg_list[i])] = bct[i]
        BCV[int(seg_num[i]):int(seg_num[i]+seg_list[i])] = bcv[i]
    return BCT,BCV    

#Calculate the integral functions F1&F2
def F1F2(x0,y0,x,y,l,nx,ny):
    k = int(Ns) #Total no. of elements
    s = x0.size #No. of nodes
    
    #creating empty variables to store data
    A, B, E, F1, F2 = [np.zeros((k,s)) for i in range(5)]
    k = np.arange(k)
    s = np.arange(s)
    K, S = np.meshgrid(k,s)
    
    A[K,:] = np.square(l[K]).T
    B[K,S] = 2*l[K]*(-(x[K]-x0[S])*ny[K]+(y[K]-y0[S])*nx[K])
    E[K,S] = (x[K]-x0[S])**2 + (y[K]-y0[S])**2
    
    M=4*A*E-B**2
    D=0.5*B/A
    zero=1e-10 #a small number to take care of error for solving
    I,J=np.where(M<zero) #Imtersecting with the element
    i,j=np.where(M>zero)#Is not intersecting with the element
    #for M=0
    F1[I,J]=0.5*l[I]*(np.log(l[I])               + (1+D[I,J])*np.log(np.abs(1+D[I,J])+zero)                  -D[I,J]*np.log(np.abs(D[I,J]+zero))-1)/np.pi
    #For M>0
    H=np.arctan((2*A[i,j]+B[i,j])/np.sqrt(M[i,j]))-np.arctan(B[i,j]/np.sqrt(M[i,j]))
    
    F1[i,j]=0.25*l[i]*(2*(np.log(l[i])-1)                  -D[i,j]*np.log(np.abs(E[i,j]/A[i,j]))                         +(1+D[i,j])*np.log(np.abs(1+2*D[i,j]+E[i,j]/A[i,j]))                           +H*np.sqrt(M[i,j])/A[i,j])/np.pi
    F2[i,j]=l[i]*(nx[i]*(x[i]-x0[j])+ny[i]*(y[i]-y0[j]))*H/np.sqrt(M[i,j])/np.pi
    
    return F1.T,F2.T

#FLOW
def pqBC(F1,F2,BCT,BCV):
    Ns=BCT.size
    F2x=F2-0.5*np.eye(Ns)
    a,b=[np.zeros((Ns,Ns)) for i in range(2)]
    
    #phi is known and -dphi/dn is unknown
    col_p = np.where(BCT==0)
    a[:,col_p]=-F1[:,col_p]
    b[:,col_p]=-F2x[:,col_p]
    
    #dphi/dn is known and - phi is unknown
    col_q=np.where(BCT==1)
    a[:,col_q]=F2x[:,col_q]
    b[:,col_q]=F1[:,col_q]
    
    BCV2 = np.linalg.solve(a,np.dot(b,BCV))
    
    p=BCV2.copy()
    q=BCV2.copy()
    
    p[col_p]=BCV[col_p] #Replace with phi
    q[col_q]=BCV[col_q] #Replace with dphi/dn
    return p,q
    
    


# # Geometry

# In[3]:


x_list = np.array([0,1,2,3,3,2,1,0,0])
y_list = np.array([-1,-1,-0.5,-0.5,0.5,0.5,1,1,-1])

#plotting the geometry
fig=plt.figure(figsize=(8,8),dpi=100)
plt.plot(x_list,y_list)


# # meshing

# In[4]:


#Boundary element method meshing values
seg_list = np.array([10,10,10,10,10,10,10,10])

#performing boundary element method
x, y, xm, ym, xms, yms, nx, ny, l, Ns, seg_num, lb = geometry(x_list,y_list,seg_list)

fig = plt.figure(figsize=(8,8),dpi=100)
fig.add_subplot(111,aspect="equal")
plt.scatter(x,y,c="r")
fig=plt.figure(figsize=(8,8),dpi=100)
fig.add_subplot(211,aspect="equal")
plt.scatter(xm,ym,c=u'g',marker=u'^')
plt.quiver(xm,ym,nx,ny)
plt.gca().set_xlim([-1,4])
plt.gca().set_ylim([-1.5,1.5])


# In[ ]:





# # boundary conditions

# In[20]:


Q=1 #volumetric flow rate
bct=np.ones(seg_list.size)  #Boundary condition type
bcv=np.zeros(seg_list.size)  #Boundary condition values
#boundary condition type
inlet=7
outlet=3
#Geometry
bcv[inlet]=-Q/lb[inlet]
bcv[outlet]=Q/lb[outlet]
#boundary condition for each and every element
BCT,BCV = SetBc(bct,bcv,seg_list,seg_num)

F1,F2 = F1F2(xm,ym,x,y,l,nx,ny)
p,q = pqBC(F1,F2,BCT,BCV)
#print(F1,F2)
#Generating the internal nodes(excludes the boundary)
Nx=20;Ny=10;

X=np.linspace(x.min(),x.max(),Nx+2)
Y=np.linspace(y.min(),y.max(),Ny+2)
X,Y=np.meshgrid(X,Y)

#Converging the section
cb=1 #Start of convergence
ce=2 #End of convergence
R=np.abs(y_list).min()/np.abs(y_list).max()

for i in range(Nx+2):
    #converging area
    if(X[0,i] > x_list[cb]) and (X[0,i] < x_list[ce]):
        m = (1-R)/(x_list[cb]-x_list[ce])
        f = 1+m*(X[0,i]-x_list[cb])
        print(f)
        Y[:,i]=Y[:,i]*f
    #After the converging area
    if(X[0,i] >= x_list[ce]):
        Y[:,i] = Y[:,i]*R
X=X[1:-1,1:-1].ravel();Y=Y[1:-1,1:-1].ravel()
plt.scatter(X,Y)


# In[21]:


#Calculating the p&q values for the internal points
delta_x=delta_y=0.05
F1,F2=F1F2((X),Y,x,y,l,nx,ny)
phi_x_plus=(np.dot(F2,p)-np.dot(F1,q)).reshape(Nx,Ny)
F1,F2=F1F2(X-delta_x,Y,x,y,l,nx,ny)
phi_x_minus=(np.dot(F2,p)-np.dot(F1,q)).reshape(Nx,Ny)
F1,F2=F1F2(X,(Y+delta_y),x,y,l,nx,ny)
phi_y_plus=(np.dot(F2,p)-np.dot(F1,q)).reshape(Nx,Ny)
F1,F2=F1F2(X,Y-delta_y,x,y,l,nx,ny)
phi_y_minus=(np.dot(F2,p)-np.dot(F1,q)).reshape(Nx,Ny)

#central difference in the velocity
u = 0.5*(phi_x_plus-phi_x_minus)/delta_x
v = 0.5*(phi_y_plus-phi_y_minus)/delta_y

#static pressure
p=-0.5*(u*u+v*v)
plt.plot(x_list,y_list)
plt.scatter(X,Y,c=u'r',marker=u'*')


# # Post processing

# In[22]:


fig=plt.figure(figsize=(8,8),dpi=100)
fig.add_subplot(111,aspect="equal")
plt.fill(x,y,fill = False,lw= 3)
plt.contourf(X.reshape(Nx,Ny),Y.reshape(Nx,Ny),p,15,alpha=0.5)
plt.colorbar()
plt.quiver(X,Y,u,v)


# In[ ]:




