#! /usr/bin/env python
from numpy import *
from fem3d import fem3d
import sys
from pysparse import itsolvers
def list_prod(*args):
    " returns the product of an arbitrary number of lists"
    ndim=len(args)
    tmp= [(i,"args[%d]" % i) for i in range(ndim)]
    fs=range(ndim)+list(reduce(lambda x,y:x+y,tmp))
    fs=tuple(fs)
    ex=("[["+ndim*"x%d,"+"]"+ndim*"for x%d in %s "+"]") %  fs
    return eval(ex)
NGL=6
xgl_list=[0.033765242898423989, 0.16939530676686773, 0.38069040695840156,
          0.61930959304159849, 0.83060469323313224, 0.96623475710157603]
wgl_list=[0.085662246189581431, 0.1803807865240693, 0.23395696728634469,
          0.23395696728634469, 0.1803807865240693, 0.085662246189581431]
xgl=array(xgl_list)
wgl=array(wgl_list)

def NodesToGLGrid(xi):
    tmp1=[]
    tmp2=[]
    hi=xi[1:]-xi[:-1]
    for x,h in zip(xi[:-1],hi):
        tmp1=tmp1+(x+xgl*h).tolist()
        tmp2=tmp2+(wgl*h).tolist()
    return tmp1,tmp2
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
    # Calculate global Gauss Legendre Grid
    xgl,wgl=NodesToGLGrid(x)
    gp=array(list_prod(xgl,xgl,xgl))
    tmp=array(list_prod(wgl,wgl,wgl))
    gwts=array([xw*yw*zw for xw,yw,zw in zip(tmp[:,0],tmp[:,1],tmp[:,2])])
    # 
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
    N1=K.shape[0] # Order of matrix without boundary DOF
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
    # convert K to sparse skyline format
    K1=K.to_sss()
    # convert U to sparse skyline format
    M=U.to_sss()
    b=empty(N0)
    # calculate U r on whole domain including boundary 
    M.matvec(rho,b)
    b=b-K0bcv
    # project on interior space yielding RHS of (8)
    b1=array([ b[i] for i in range(N0) if mask[i] == 1 ])
    pot=empty(N1)
    # specify parameters for iterative solver 
    maxit=20000
    tol=1e-15
    # call qmrs from pysparse.itsolvers 
    t0=time()
    info,iter,relres=itsolvers.qmrs(K1,b1,pot,tol,maxit)
    t1=time()
    print info,iter,relres
    print "time taken to solve Laplace equation using LU Decomposition :", t1-t0
    pot0=empty(N0)
    # the following code fills the vector pot0 with the boundary values
    # of the potential at the corresponding positions
    for i in range(N0):
        if mask[i] ==0:
            x,y,z=nodes[i]
            pot0[i]=bval(x,y,z)
    pot0[ii]=pot
    def potf(xi,yi,zi):
        return box.wave(xi,yi,zi,pot0)
    def gradpotf(xi,yi,zi):
        return box.gradwave(xi,yi,zi,pot0)
    # first plot resulting pot along a line 
    xi=linspace(-xmax,xmax,1001)
    yi=zi=zeros(1001,"i")
    poti=potf(xi,yi,zi)
    dpoti=poti-array([exactpot(xx,0,0) for xx in xi])
    hx=2*xmax/1000.0
    errsq=2*hx*pi*sum(xi**2*dpoti**2)
    err=sqrt(errsq)
    # the following evaluates the potential at all points of the
    # Gauss-Legendre Grid and also evaluates the difference
    # to the exact solution as well as the norm of the error 
    pot3atgp=potf(gp[:,0],gp[:,1],gp[:,2])
    dpot3atgp=pot3atgp-array([exactpot(xx,yy,zz) for xx,yy,zz in zip(gp[:,0],gp[:,1],gp[:,2])])
    errsq3d=sum(gwts*dpot3atgp**2)
    err3d=sqrt(errsq3d)
    # Test gradient at point (1.0,1.0,1.0) and compare with grad of exact solution
    print "GRADIENT at 1,1,1:",gradpotf([1.0,],[1.0,],[1.0,]),gradexactpot(1.0,1.0,1.0)
    # the same as above but now for the gradient of the potential in comparison to the
    # gradient of the exact solution 
    tmp1=array(gradpotf(gp[:,0],gp[:,1],gp[:,2]))
    tmp2=array(gradexactpot(gp[:,0],gp[:,1],gp[:,2]))
    tmp=tmp1-tmp2
    errgradsq=sum(gwts*(tmp[0]**2+tmp[1]**2+tmp[2]**2))
    errgrad=sqrt(errgradsq)
    # open file to which the 1D plot is written 
    pfile=open("pot_xmax=%f_nel=%d_no=%d.dat" %  (xmax,2*nelx,no), "w")
    print >> pfile, "#","N0=",N0, "N1=",N1, "ERR=",err ,"ERR3=",err3d, "ERRGRAD=",errgrad
    for x,p,dp in zip(xi,poti,dpoti):
        print >> pfile, x,p,dp
    pfile.close()
    # now prepare plot in the plane z=0
    xi=yi=linspace(-xmax,xmax,129)
    xyz=array(list_prod(xi,yi,[0.0,]))
    poti2d=potf(xyz[:,0],xyz[:,1],xyz[:,2])
    dpoti2d=poti2d-array([exactpot(xx,yy,0) for xx,yy in zip(xyz[:,0],xyz[:,1])])
    tmp1=array(gradpotf(xyz[:,0],xyz[:,1],xyz[:,2]))
    tmp2=array(gradexactpot(xyz[:,0],xyz[:,1],xyz[:,2]))
    tmp=tmp1-tmp2
    dgrad2d=tmp[0]**2+tmp[1]**2+tmp[2]**2
    # open file to which the surface  plot is written 
    pfile=open("pot2d_xmax=%f_nel=%d_no=%d.dat" %  (xmax,2*nelx,no), "w")
    print >> pfile, "#","N0=",N0, "N1=",N1
    i=0
    for p,pot,dp,dg in zip(xyz,poti2d,dpoti2d,dgrad2d):
        x,y,z=p
        print >> pfile, x,y,pot,pot-exactpot(x,y,z),dp,dg
        i=i+1
        if i % 129 == 0:
            print >> pfile
    pfile.close()
def dens(x,y,z):
    return 4*exp(-2*sqrt(x*x+y*y+z*z))
def exactpot(x,y,z):
    r=sqrt(x*x+y*y+z*z)
    if r > 0:
        return (1-(1+r)*exp(-2*r))/r
    else:
        return 1.0
def gradexactpot(x,y,z):
    r=sqrt(x*x+y*y+z*z)
    gr=(2*(r+1)*exp(-2*r)-exp(-2*r))/r+((r+1)*exp(-2*r)-1)/r/r
    return gr*array([x/r,y/r,z/r])
# main program
# read rmax, nel and no as command line arguments
rmax,nel,no=map(eval,sys.argv[1:4])
if nel % 2 != 0:
    print "nel must be even!"
    sys.exit()
nelx=nel/2.0
PoissonSolveFem(rmax,nelx,no,exactpot,dens)

