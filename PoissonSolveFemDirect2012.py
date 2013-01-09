#! /usr/bin/env python
from numpy import *
from fem3d import fem3d
import sys
from time import time
from pysparse import superlu
def PoissonSolveFem(xmax,nelx,no,bval,dens):
    bdcon=array([1,1,1,1,1,1]) # these bc's to for non zero Dirichlet   
    tx=linspace(0.0,1.0,nelx+1)
    # nodes quadratically scaled around origin!
    t2=tx**2
    tm=(-t2).tolist()
    tm.reverse()
    t=array(tm[:-1]+t2.tolist())
    print t
    x=y=z=t*xmax
    print x
    print y
    print z
    box=fem3d(x,y,z,no,bdcon)
    def v(xx,yy,zz):
        val=0.0
        return val
    def w(xx,yy,zz):
        return 1.0
    def IsOnBoundary(x,y,z):
        # this function returns True if (x,y,z) is on the Boundary
        if (abs(x) == xmax or abs(y) ==xmax or abs(z) == xmax):
            val= True
        else:
            val= False
        return val
    def bval1(x,y,z):
        # returns  phi (x,y,z) for (x,y,z) on the boundary 
        # but otherwise zero 
        if IsOnBoundary(x,y,z):
            return bval(x,y,z)
        else:
            return 0.0
    box.calc_mat(v,w)
    nodes=box.gn
    # 
    K=box.All # Matrix representing negative Laplace operator
    U=box.Mll # Overlap matrix between basis functions
    N0=K.shape[0] # Order of matrix including boundary DOF
    # rho is density at points of FEM grid
    rho=array([dens(x,y,z) for (x,y,z) in nodes])
    # mask is needed for pysparse 
    mask=array([1-IsOnBoundary(x,y,z) for (x,y,z) in nodes]) 
    bcv=array([bval1(x,y,z) for (x,y,z) in nodes])
    K0bcv=empty(N0)
    # convert K to sparse skyline format
    K0=K.to_sss()
    # calculate ( K_{01}+K_{11}) c_1 
    K0.matvec(bcv,K0bcv)
    # modify matrix K to implement Boundary Condition
    # all rows and columns corresponding to boundary nodes are deleted from the matrix 
    K.delete_rowcols(mask) 
    N1=K.shape[0] # Order of matrix without boundary dof's
    print K.shape
    # create new list of global nodes as 
    newnodes=array([nodes[i] for i in range(N0) if mask[i] ==1 ])
    # create indices for new nodes in old nodes
    ii=zeros(N1,"i")
    j=0
    for i in range(N0):
        if mask[i] == 1:
            ii[j]=i
            j=j+1
    # convert K to compressed sparse row format 
    K1=K.to_csr()
    # convert U to sparse skyline format 
    M=U.to_sss()
    b=empty(N0)
    # calculate U rho on whole domain including boundary
    M.matvec(rho,b)
    b=b-K0bcv
    # project on interior space yielding RHS of (8)
    b1=array([ b[i] for i in range(N0) if mask[i] == 1 ]) 
    pot=empty(N1)
    t0=time()
    # calculate LU factorization using the superlu module from pysparse
    LU=superlu.factorize(K1)  
    t1=time()
    print "time taken for factorization of K1:", t1-t0
    t0=time()
    LU.solve(b1,pot)
    t1=time()
    print "time taken to solve Laplace equation using LU Decomposition :", t1-t0
    pot0=empty(N0)
    # the following code fills the vector pot0 with the boundary values
    # of the potential at the corresponding positions
    for i in range(N0):
        if mask[i] ==0:
            x,y,z=nodes[i]
            pot0[i]=bval(x,y,z)
    pot0[ii]=pot
    # making use of the interpolation functionality provided by fem3d
    # define a function that returns the values at of the resulting potential
    # at the points defined by (xi[k],yi[k],zi[k])
    def potf(xi,yi,zi):
        return box.wave(xi,yi,zi,pot0)
    # the following lines define xi,yi,zi along a line with constant y and z-values
    # from -xmax to xmax 
    xi=linspace(-xmax,xmax,1001)
    yi=zi=zeros(1001,"i")
    poti=potf(xi,yi,zi)
    #open file to write potential values to
    pfile=open("pot_direct_xmax=%f_nel=%d_no=%d.dat" %  (xmax,2*nelx,no), "w")
    print >> pfile, "# N0=",N0, "N1=",N1
    for x,p in zip(xi,poti):
        print >> pfile, x,p,p-exactpot(x,0,0)
    pfile.close()
def dens(x,y,z):
    return 4*exp(-2*sqrt(x*x+y*y+z*z))
def exactpot(x,y,z):
    r=sqrt(x*x+y*y+z*z)
    if r > 0:
        return (1-(1+r)*exp(-2*r))/r
    else:
        return 1.0
# main program
# read rmax, nel and no as command line arguments
rmax,nel,no=map(eval,sys.argv[1:4])
if nel % 2 != 0:
    print "nel must be even!"
    sys.exit()
nelx=nel/2.0 
PoissonSolveFem(rmax,nelx,no,exactpot,dens)

