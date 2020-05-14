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




nvec = [1,2,3,4,5,6,7,8,9,10,20,30,50,100,1e3,1e4,1e5,1e6]
uvec = []
uLinfvec = []
wLinfvec = []

for idx, n in enumerate(nvec):
    uvec.append(problem(V,n))
    if idx == 0:
        continue
    uLinfvec.append(compute_Linf(uvec[idx], uvec[idx-1]))
    wLinfvec.append(compute_Linf(w_n(nvec[idx]), w_n(nvec[idx-1])))

#File(f'u_n={n}.pvd') << u



x_equals_y = np.linspace(0,max(wLinfvec),10000)

# normal 
ax0 = plt.subplot(1,2,1)
ax0.plot(wLinfvec, uLinfvec, 'o')
#ax0.plot(x_equals_y, x_equals_y, ':')
ax0.set_xlabel('$||w^{(n+1)}-w^{(n)}||_{L_\infty}$')
ax0.set_ylabel('$||u^{(n+1)}-u^{(n)}||_{L_\infty}$')
slope, intercept, r_value, p_value, std_err = stats.linregress(wLinfvec, uLinfvec)
text0 = f"slope = {slope:.4f}\nintercept = {intercept:.4f}\n$r^2$ = {(r_value**2):.4f}"
ax0.text(0.2,0.8,text0,horizontalalignment='center', 
         verticalalignment='center', transform=ax0.transAxes)

# log
ax1 = plt.subplot(1,2,2)
ax1.plot(np.log10(wLinfvec), np.log10(uLinfvec), 'o')
ax1.set_xlabel('$log_{10} ||w^{(n+1)}-w^{(n)}||_{L_\infty}$')
ax1.set_ylabel('$log_{10} ||u^{(n+1)}-u^{(n)}||_{L_\infty}$')
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(wLinfvec), np.log10(uLinfvec))
text1 = f"slope = {slope:.4f}\nintercept = {intercept:.4f}\n$r^2$ = {(r_value**2):.4f}"
ax1.text(0.2,0.8,text1,horizontalalignment='center', 
         verticalalignment='center', transform=ax1.transAxes)
#ax1.plot(np.log10(x_equals_y), np.log10(x_equals_y), ':')

plt.show()

