from dolfin import *
import ufl
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pygmsh

# Load in spherical mesh
comm = MPI.comm_world
m = UnitDiscMesh.create(comm, 50, 1, 3)

x = MeshCoordinates(m)
dx = Measure("dx", m)
class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > (xmax-DOLFIN_EPS)

def boundary(x, on_boundary):
    return on_boundary # x[0] > 0.99

# functions
V = FunctionSpace(m,"CG",1)

def w_n(n):
    return project((1+x[0]+x[1])/n, V)

# compute Linf norm of absolute difference in solutions
def compute_Linf(u_0, u_1):
    return max(abs(u_0.compute_vertex_values() - u_1.compute_vertex_values()))


# problem
def problem(V, n):
    u = Function(V)
    v = TestFunction(V)
    g = lambda n: Expression(f"(1+x[0]+x[1])/{n}", degree=1)
    #bc = DirichletBC(V, Constant(0.0), boundary)
    bc = DirichletBC(V, g(n), boundary)
    
    F = inner(grad(u), grad(v))*dx + ufl.sinh(u)*v*dx

    solve(F==0, u, bc)
    File(f'u_n={n}.pvd') << u

    return u




nvec = [1,3,10,30,1e2,3e2,1e3,3e3,1e4]
uvec = []
uLinfvec = []
wLinfvec = []

for idx, n in enumerate(nvec):
    uvec.append(problem(V,n))
    if idx == 0:
        continue
    uLinfvec.append(compute_Linf(uvec[idx], uvec[idx-1]))
    wLinfvec.append(compute_Linf(w_n(nvec[idx]), w_n(nvec[idx-1])))

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


