from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

m = Mesh("unit_ball.xml")
mb = BoundaryMesh(m, "exterior")
#translation = Point((-0.5,-0.5))
#m.translate(translation)
x = MeshCoordinates(m)
dx = Measure("dx", m)
dxb = Measure("dx", mb)

def boundary(x, on_boundary):
    return on_boundary

# parameters
k0 = 1
k1 = 1

# functions
V = FunctionSpace(m,"CG",1)
Vp = VectorFunctionSpace(m,"CG",1,dim=2) # Vb projected onto m

Vb = VectorFunctionSpace(mb,"CG",1,dim=2)
Vpb = FunctionSpace(mb,"CG",1) # V projected onto mb
#V0 = V.sub(0).collapse()

def w_n(n):
    #return (x[0]+x[1])/n
    return project((1+x[0]+x[1]+x[2])/n, V)
    #return project((x[0]+x[1]+x[2])/n, V0)
    #return project(Constant(0), V0)

# compute Linf norm of absolute difference in solutions
def compute_Linf(u_0, u_1):
    return max(abs(u_0.compute_vertex_values() - u_1.compute_vertex_values()))

# problem
def problem(n):
    u = Function(V)
    up = Function(Vp)
    u_ = project(Constant(0), V)

    ub = Function(Vb)
    upb = Function(Vpb)
    ub_ = project(as_vector([Constant(0), Constant(0)]), Vb)
    wp_n = Function(Vpb)

    LagrangeInterpolator.interpolate(up, ub)
    LagrangeInterpolator.interpolate(upb, u)
    LagrangeInterpolator.interpolate(wp_n, w_n(n))

    v = TestFunction(V)
    vb = TestFunction(Vb)

    dt = 0.01
    t = dt
    bc = []
    
    jf = k0*u*ub[0]
    jr = k1*ub[1]
    j = jf - jr


    #j = k0*u*up[0] - k1*up[1]
    #jb = k0*upb*ub[0] - k1*ub[1]
    #F0 = inner((u-u_)/dt, v)*dx + inner(grad(u), grad(v))*dx - (- j + w_n(n))*v*ds
    #F1 = inner((ub-ub_)/dt, vb)*dxb + inner(grad(ub), grad(vb))*dxb - (-jb*vb[0] + jb*vb[1] + wp_n*(vb[0]+vb[1]))*dxb

    j = k0*u*(up[0]+w_n(n)) - k1*(up[1]+w_n(n))
    jb = k0*upb*(ub[0]+wp_n) - k1*(ub[1]+wp_n)
    #j = k0*u*(sinh+w_n(n)) - k1*(up[1]+w_n(n))
    #jb = k0*upb*(ub[0]+wp_n) - k1*(ub[1]+wp_n)
    F0 = inner((u-u_)/dt, v)*dx + inner(grad(u), grad(v))*dx - (- j + w_n(n))*v*ds
    F1 = inner((ub-ub_)/dt, vb)*dxb + inner(grad(ub), grad(vb))*dxb - (-jb*vb[0] + jb*vb[1] + wp_n*(vb[0]+vb[1]))*dxb

    vtkfile = File(f'u_n={n}_.pvd')
    vtkfileb = File(f'ub_n={n}_.pvd')

    while t < 0.1:
        solve(F0==0, u, bc)
        solve(F1==0, ub, bc)

        LagrangeInterpolator.interpolate(up, ub)
        LagrangeInterpolator.interpolate(upb, u)
        LagrangeInterpolator.interpolate(wp_n, w_n(n))

        t += dt
        u_.assign(u)
        ub_.assign(ub)
        #vtkfile << (u,t)
        #vtkfileb << (ub,t)

    return u, ub

#nvec = [1,3,10,30,1e2,3e2,1e3]
nvec = [1,3,10,30,1e2,3e2,1e3,3e3,1e4]
#nvec = [1,3]
uvec = []
ubvec = []
uLinfvec = []
ubLinfvec = []
utotLinfvec = []
wLinfvec = []

for idx, n in enumerate(nvec):
    u, ub = problem(n)
    uvec.append(u)
    ubvec.append(ub)
    if idx == 0:
        continue
    uLinfvec.append(compute_Linf(uvec[idx], uvec[idx-1]))
    ubLinfvec.append(compute_Linf(ubvec[idx], ubvec[idx-1]))
    utotLinfvec.append(max(uLinfvec[-1], ubLinfvec[-1]))
    wLinfvec.append(compute_Linf(w_n(nvec[idx]), w_n(nvec[idx-1])))

x_equals_y = np.linspace(0,max(wLinfvec),10000)


uplot = utotLinfvec
# normal 
ax0 = plt.subplot(1,2,1)
ax0.plot(wLinfvec, uplot, 'o')
#ax0.plot(x_equals_y, x_equals_y, ':')
ax0.set_xlabel('$||w^{(n+1)}-w^{(n)}||_{L_\infty}$')
ax0.set_ylabel('$||u^{(n+1)}-u^{(n)}||_{L_\infty}$')
slope, intercept, r_value, p_value, std_err = stats.linregress(wLinfvec, uplot)
text0 = f"slope = {slope:.4f}\nintercept = {intercept:.4f}\n$r^2$ = {(r_value**2):.4f}"
ax0.text(0.2,0.8,text0,horizontalalignment='center', 
         verticalalignment='center', transform=ax0.transAxes)


# log
ax1 = plt.subplot(1,2,2)
ax1.plot(np.log10(wLinfvec), np.log10(uplot), 'o')
ax1.set_xlabel('$log_{10} ||w^{(n+1)}-w^{(n)}||_{L_\infty}$')
ax1.set_ylabel('$log_{10} ||u^{(n+1)}-u^{(n)}||_{L_\infty}$')
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(wLinfvec), np.log10(uplot))
text1 = f"slope = {slope:.4f}\nintercept = {intercept:.4f}\n$r^2$ = {(r_value**2):.4f}"
ax1.text(0.2,0.8,text1,horizontalalignment='center', 
         verticalalignment='center', transform=ax1.transAxes)
#ax1.plot(np.log10(x_equals_y), np.log10(x_equals_y), ':')

plt.show()

#slope, intercept, r_value, p_value, std_err = stats.linregress(wLinfvec, uLinfvec)





