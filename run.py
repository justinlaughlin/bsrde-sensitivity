from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pygmsh

# Here we use pygmsh to auto generate meshes
g = pygmsh.built_in.Geometry()
g.add_ball([0,0,0], 1.0, lcar=0.1) # lcar is characteristic length
m = pygmsh.generate_mesh(g, prune_vertices=False)


# Load in spherical mesh
vm = Mesh('sphere.xml')
m = BoundaryMesh(vm, 'exterior')
xmax_idx = np.argmax(m.coordinates()[:,0])
xmax = m.coordinates()[xmax_idx,0]

x = MeshCoordinates(m)
dx = Measure("dx", m)
mf = MeshFunction('size_t', m, 0, 0)
class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > (xmax-DOLFIN_EPS)

# mark bc
#boundary = Boundary()
#boundary.mark(mf, 4)
#assert len(mf.where_equal(4)) == 1
def boundary(x, on_boundary):
    return x[0] > 0.99

# functions
V = FunctionSpace(m,"CG",1)

def f(x):
    return 2*x[0]

# compute L2 norm of error
def compute_L2_err(uex, u):
    return sqrt(assemble((u-uex)**2*dx))

# problem
u = TrialFunction(V)
v = TestFunction(V)
bc = DirichletBC(V,x[0], boundary)

a = inner(grad(u), grad(v))*dx
L = f(x)*v*dx

u = Function(V)
    
solve(a==L, u, bc)
File(f'u_n={n}.pvd') << u



