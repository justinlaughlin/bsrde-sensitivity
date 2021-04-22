from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pygmsh
import random

# Here we use pygmsh to auto generate meshes
#g = pygmsh.built_in.Geometry()
#g.add_ball([0,0,0], 1.0, lcar=0.1) # lcar is characteristic length
#m = pygmsh.generate_mesh(g, prune_vertices=False)

# Load in spherical mesh
vm = Mesh('sphere.xml')
m = BoundaryMesh(vm, 'exterior')
N = m.coordinates().shape[0]

x = MeshCoordinates(m)
dx = Measure("dx", m)
mf = MeshFunction('size_t', m, 0, 0)

V = VectorFunctionSpace(m,"CG",1,dim=3)
V0 = V.sub(0).collapse()

#def w_n(n):
#    return project((1 + x[0] + x[1] + x[2] + x[0]*x[1] + x[0]*x[2] + x[1]*x[2] + x[0]**2 + x[1]**2 + x[2]**2)/n, V0)
def w_n(n):
    return project((1 + x[0] + x[1] + x[2])**10/n, V0)

# compute Linf norm of absolute difference in solutions
def compute_Linf(u_0, u_1):
    return max(abs(u_0.compute_vertex_values() - u_1.compute_vertex_values()))


def rng_bc(r):
    rx, ry, rz = m.coordinates()[r,:]*0.90

    def rng_boundary(x, on_boundary):
        return x[0] > rx and x[1] > ry and x[2] > rz 
    print(f"Randomly selected point has coordinates: {rx}, {ry}, {rz}")

    return rng_boundary


# problem
def problem(V, n): 
    # parameters
    k0 = 1
    k1 = 1
    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V,as_vector(rand_dirichlet), rng_boundary)
    
    jf = k0*u[0]*u[1]
    jr = k1*u[2]
    j = jf - jr
    #F0 = inner(grad(u), grad(v))*dx - (-j*v[0] + -j*v[1] + j*v[2] + w_n(n)*(v[0]+v[1]+v[2]))*dx
    F0 = inner(grad(u), grad(v))*dx - (-j*v[0] + -j*v[1] + j*v[2])*w_n(n)*dx

    solve(F0==0, u, bc)
    File(f'u_n={n}.pvd') << u

    return u

# note: we can test x dependent dirichlet 

# # Run with many different bcs
# for j in range(10):
#     rand_dirichlet = [random.random(), random.random(), random.random()]
#     rng_boundary = rng_bc()
#     u = problem(V,n)
#     File(f'u_{n}_{j}.pvd') << u


nvec = [1e5, 1e7, 1e9, 1e11]
uvec = []
uLinfvec = []
wLinfvec = []


rand_dirichlet = [random.random(), random.random(), random.random()]
r = int(np.round(N*random.random())) # a randomly selected point
rng_boundary = rng_bc(r)
for idx, n in enumerate(nvec):
    uvec.append(problem(V,n))
    if idx == 0:
        continue
    uLinfvec.append(compute_Linf(uvec[idx], uvec[idx-1]))
    wLinfvec.append(compute_Linf(w_n(nvec[idx]), w_n(nvec[idx-1])))


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

