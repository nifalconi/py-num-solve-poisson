#!/usr/bin/env python
from numpy import *
from fem1d import fem1d
from pysparse import spmatrix
import sys
from time import time
# read command line arguments
args=sys.argv
xmax,nel,no=map(eval,args[1:4])
nel1=nel/2
# read functions as strings used to define density, potential and boundary condition 
srho,sbc,spot=args[4:7]
def rho(x,y,z):
    r2=x*x+y*y+z*z
    r=sqrt(r2)
    return eval(srho)
def pot(x,y,z):
    r2=x*x+y*y+z*z
    r=sqrt(r2)
    return eval(spot)
def bc(x,y,z): 
    r2=x*x+y*y+z*z
    r=sqrt(r2)    
    return eval(sbc)
# create quadratically scaled 1D grid 
t=linspace(0.0,1.0,nel1+1)
xi=t**2*xmax
xm=(-xi).tolist()
xm.reverse()
xel=array(xm[:-1]+xi.tolist())
ngl=12
# calculate matrices and eigen pairs for 1D finite element problem of particle 
# in box [-x_max ,x_max]. hm01 and um01 are the kinetic energy and overlap matrices
# between inside functions and the two  boundary functions. 
# only one set of matrices needed, since the grid is the same for x,y and z
t1=time()
nmat,evals,evmat,umat,hmat,xn,hm01,um01=fem1d(xel,no,ngl,0,bc=True)
t2=time()
print "Time for solution of 1D eigen problem=",t2-t1
evmatT=transpose(evmat)
nmat3=nmat**3
print "nmat3=",nmat3
# this function multiplies v  by tensor product of u_x,u_y, u_z, all equal to u
def umul3(v):
    u=v.copy()
    u.shape=nmat,nmat,nmat
    for i in range(nmat):
        for j in range(nmat):
            tmp=u[i,j].copy()
            u[i,j]=dot(umat,tmp)
    for i in range(nmat):
        for k in range(nmat):
            tmp=u[i,:,k].copy()
            u[i,:,k]=dot(umat,tmp)
    for j in range(nmat):
        for k in range(nmat):
            tmp=u[:,j,k].copy()
            u[:,j,k]=dot(umat,tmp)
    u.shape=nmat3
    return u
# this function multiplies a vector v  by tensor product of evmatT in x,y,z 
def h0tmul(v): 
    u=v.copy()
    u.shape=nmat,nmat,nmat
    for i in range(nmat):
        for j in range(nmat):
            tmp=u[i,j].copy()
            u[i,j]=dot(evmatT,tmp)
    for i in range(nmat):
        for k in range(nmat):
            tmp=u[i,:,k].copy()
            u[i,:,k]=dot(evmatT,tmp)
    for j in range(nmat):
        for k in range(nmat):
            tmp=u[:,j,k].copy()
            u[:,j,k]=dot(evmatT,tmp)
    u.shape=nmat3
    return u
# this function multiplies a vector v  by tensor product of evmat in x,y,z 
def h0mul(v): 
    u=v.copy()
    u.shape=nmat,nmat,nmat
    for i in range(nmat):
        for j in range(nmat):
            tmp=u[i,j].copy()
            u[i,j]=dot(evmat,tmp)
    for i in range(nmat):
        for k in range(nmat):
            tmp=u[i,:,k].copy()
            u[i,:,k]=dot(evmat,tmp)
    for j in range(nmat):
        for k in range(nmat):
            tmp=u[:,j,k].copy()
            u[:,j,k]=dot(evmat,tmp)
    u.shape=nmat3
    return u
# U-norm for vectors in 3D space 
def unorm(v):	
    return sqrt(dot(umul3(v),v))
# the following creates 3 arrays xi,yi and zi such that 
# [xi[k],yi[k],zi[k]] is the k-th point on the 3D grid 
xi=zeros((nmat,nmat,nmat),"d")
yi=zeros((nmat,nmat,nmat),"d")
zi=zeros((nmat,nmat,nmat),"d")
for i in range(nmat):
    x=xn[i].copy()
    for j in range(nmat):
        y=xn[j].copy()
        for k in range(nmat):
            z=xn[k].copy()
            xi[i,j,k]=x
            yi[i,j,k]=y
            zi[i,j,k]=z
xi.shape=nmat3
yi.shape=nmat3
zi.shape=nmat3

t1=time()
evals3=zeros((nmat,nmat,nmat),"d")
for i in range(nmat):
    for j in range(nmat):
        for k in range(nmat):
            evals3[i,j,k]=evals[i]+evals[j]+evals[k]
evals3.shape=nmat3
gf=1.0/evals3
# the next section creates the matrix required for solving the Poisson equation 
# with a boundary condition given by a function bc(x,y,z)
# create dictionary versions of hm01 and um01
um01d,hm01d,hmatd,umatd={},{},{},{} 
for i in range(nmat):
    for j in range(2):
        if um01[i,j] != 0.0:
            um01d[i,j]=um01[i,j]
        if hm01[i,j] != 0.0:
            hm01d[i,j]=hm01[i,j]
for i in range(nmat):
    for j in range(nmat):
        if umat[i,j] != 0.0:
            umatd[i,j]=umat[i,j]
        if hmat[i,j] != 0.0:
            hmatd[i,j]=hmat[i,j]
# create list of boundary points
xnb=array([-xmax,]+xn.tolist()+[xmax])
pbi=[]
bdict={}
i=0
for xx in xnb:
    for yy in xnb:
        for zz in xnb:
            if abs(xx) == xmax or abs(yy) == xmax or abs(zz) == xmax:
                pbi.append((xx,yy,zz))
                bdict[(xx,yy,zz)]=i
                i=i+1
# create three arrays xbi,ybi,zbi to vectorize evaluation of bc
xbi,ybi,zbi=[],[],[]
for xx,yy,zz in pbi:
    xbi.append(xx)
    ybi.append(yy)
    zbi.append(zz)
xbi=array(xbi)
ybi=array(ybi)
zbi=array(zbi)
def bindex(xx,yy,zz):
    return bdict[(xx,yy,zz)]
nb=len(pbi) # number of boundary nodes
# create sparse matrix K01 in dictionary form 
t1=time()
k01mat=spmatrix.ll_mat(nmat3,nb)
def xx(j,r):
    if r == 0:
        return xn[j]
    else:
        return (-1+2*j)*xmax
# the following loop over r,s,t corresponds to  the three terms in (20) 
for r in [0,1]:
    for s in [0,1]:
        for t in [0,1]:
            if r+s+t > 0: # at least one of r,s,t is 1!
                print r,s,t
            #combination HUU
                dx,dy,dz=hmatd,umatd,umatd
                if r:
                    dx=hm01d
                if s:
                    dy=um01d
                if t:
                    dz=um01d
                for kx,mx in dx.items():
                    ix,jx=kx
                    xb=xx(jx,r)
                    for ky,my in dy.items():
                        iy,jy=ky
                        yb=xx(jy,s)
                        for kz,mz in dz.items():
                            iz,jz=kz
                            zb=xx(jz,t)
                            ii,jj=ix*nmat**2+iy*nmat+iz,bindex(xb,yb,zb)
                            tmp=mx*my*mz
                            k01mat[ii,jj]=k01mat[ii,jj]+tmp
            #combination UHU
                dx,dy,dz=umatd,hmatd,umatd
                if r:
                    dx=um01d
                if s:
                    dy=hm01d
                if t:
                    dz=um01d
                for kx,mx in dx.items():
                    ix,jx=kx
                    xb=xx(jx,r)
                    for ky,my in dy.items():
                        iy,jy=ky
                        yb=xx(jy,s)
                        for kz,mz in dz.items():
                            iz,jz=kz
                            zb=xx(jz,t)
                            ii,jj=ix*nmat**2+iy*nmat+iz,bindex(xb,yb,zb)
                            tmp=mx*my*mz
                            k01mat[ii,jj]=k01mat[ii,jj]+tmp
            #combination UUH
                dx,dy,dz=umatd,umatd,hmatd
                if r:
                    dx=um01d
                if s:
                    dy=um01d
                if t:
                    dz=hm01d
                for kx,mx in dx.items():
                    ix,jx=kx
                    xb=xx(jx,r)

                    for ky,my in dy.items():
                        iy,jy=ky
                        yb=xx(jy,s)
                        for kz,mz in dz.items():
                            iz,jz=kz
                            zb=xx(jz,t)
                            ii,jj=ix*nmat**2+iy*nmat+iz,bindex(xb,yb,zb)
                            tmp=mx*my*mz
                            k01mat[ii,jj]=k01mat[ii,jj]+tmp

t2=time()
dt=t2-t1
print "time taken for creating k01mat=",dt
# now create u01mat appearing in (21)
t1=time()
u01mat=spmatrix.ll_mat(nmat3,nb)
for r in [0,1]:
    for s in [0,1]:
        for t in [0,1]:
            if r+s+t > 0: # at least one of r,s,t is 1!
                print r,s,t
            #combination HUU
                dx,dy,dz=umatd,umatd,umatd
                if r:
                    dx=um01d
                if s:
                    dy=um01d
                if t:
                    dz=um01d
                for kx,mx in dx.items():
                    ix,jx=kx
                    xb=xx(jx,r)
                    for ky,my in dy.items():
                        iy,jy=ky
                        yb=xx(jy,s)
                        for kz,mz in dz.items():
                            iz,jz=kz
                            zb=xx(jz,t)
                            ii,jj=ix*nmat**2+iy*nmat+iz,bindex(xb,yb,zb)
                            tmp=mx*my*mz
                            u01mat[ii,jj]=u01mat[ii,jj]+tmp

t2=time()
dt=t2-t1
print "time taken for creating u01mat=",dt
def k01mul(bcvec):
    tmp=zeros(nmat3,"d")
    k01mat.matvec(bcvec,tmp)
    return -tmp
def u01mul(rhobc):
    tmp=zeros(nmat3,"d")
    u01mat.matvec(rhobc,tmp)
    return tmp
def solvePoisson(rho,bcvec,rhobc):
    return h0mul(gf*h0tmul(umul3(rho)+k01mul(bcvec)+u01mul(rhobc)))  
#######
potex=pot(xi,yi,zi)
rhoi=rho(xi,yi,zi)
t1=time()
poti=solvePoisson(4*pi*rhoi,bc(xbi,ybi,zbi),4*pi*rho(xbi,ybi,zbi))
t2=time()
dt=t2-t1
print "time taken for solution=",dt
err=poti-potex
print unorm(err) 

