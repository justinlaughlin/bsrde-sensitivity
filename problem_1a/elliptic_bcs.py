from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pygmsh

## Here we use pygmsh to auto generate meshes
#g = pygmsh.built_in.Geometry()
#g.add_ball([0,0,0], 1.0, lcar=0.1) # lcar is characteristic length
#m = pygmsh.generate_mesh(g, prune_vertices=False)


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

# parameters
k0 = 1
k1 = 1

# functions
V = VectorFunctionSpace(m,"CG",1,dim=3)
V0 = V.sub(0).collapse()

def w_n(n):
    return project((1+x[0]+x[1])/n, V0)

# compute Linf norm of absolute difference in solutions
def compute_Linf(u_0, u_1):
    return max(abs(u_0.compute_vertex_values() - u_1.compute_vertex_values()))


# problem
def problem(V, n):
    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V,as_vector([x[0],x[0],x[0]]), boundary)
    #bc = []
    
    jf = k0*u[0]*u[1]
    jr = k1*u[2]
    j = jf - jr
    F0 = inner(grad(u), grad(v))*dx - (-j*v[0] + -j*v[1] + j*v[2] + w_n(n)*(v[0]+v[1]+v[2]))*dx

    solve(F0==0, u, bc)
    File(f'u_n={n}.pvd') << u

    return u




nvec = [1,2,3,4,5,6,7,8,9,10]
uvec = []
uLinfvec = []
wLinfvec = []

for idx, n in enumerate(nvec):
    print(idx)
    uvec.append(problem(V,n))
    if idx == 0:
        continue
    uLinfvec.append(compute_Linf(uvec[idx], uvec[idx-1]))
    wLinfvec.append(compute_Linf(w_n(nvec[idx]), w_n(nvec[idx-1])))

#File(f'u_n={n}.pvd') << u



