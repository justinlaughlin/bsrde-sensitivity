from dolfin import *
import numpy as np

# generate mesh centered at (0,0)
Nseg = 60
m = UnitSquareMesh(Nseg, Nseg)
N = m.coordinates().shape[0]
translation = Point((-0.5,-0.5))
m.translate(translation)
x = MeshCoordinates(m)
mf = MeshFunction('size_t', m, 1, 0)
# class Boundary(SubDomain):
#     def __init__(self, x, on_boundary):
#         return on_boundary
# boundary = Boundary()
def boundary(x, on_boundary):
    return on_boundary

# parameters
k0 = 1
k1 = 1

# functions
V = VectorFunctionSpace(m,"CG",1,dim=3)
u = Function(V)
v = TestFunction(V)
bc = DirichletBC(V,Constant((0,0,0)), boundary)

jf = k0*u[0]*u[1]
jr = k1*u[2]
j = jf - jr

def w_n(n):
    return (x[0]+x[1])/n

for n in [1,2,3,4,5,6,8,10,15,20,25,30,50,75,1e2,1e3,1e4]:
    F0 = inner(grad(u), grad(v))*dx - (j*v[0] + j*v[1] - j*v[2] + w_n(n)*(v[0]+v[1]+v[2]))*dx
    solve(F0==0, u, bc)
    File(f'u_n={n}.pvd') << u


