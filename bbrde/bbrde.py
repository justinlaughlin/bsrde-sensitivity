from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def multiphys_loop(F_list, u_list, uinterp_list):
    error_list = [1]*len(F_list)
    abs_tol = 1e-8

    while any([e>=abs_tol for e in error_list]):
        for idx, F in enumerate(F_list):
            solve(F==0, u_list[idx], [])
            LagrangeInterpolator.interpolate(uinterp_list[idx], u_list[idx])

            # compute residual
            Fv = assemble(F)
            Fv.abs()
            error_list[idx] = norm(Fv, 'linf')
            print(f"Fabs for compartment with index {idx}: {error_list[idx]}")

    print("All Fabs are below tolernace...")    

# compute Linf norm of absolute difference in solutions
def compute_Linf(u_0, u_1):
    return max(abs(u_0.compute_vertex_values() - u_1.compute_vertex_values()))

def problem(n):
    # mesh
    mtot    = Mesh("inner_outer_balls.xml") # 
    x       = MeshCoordinates(mtot)
    mf3     = MeshFunction('size_t', mtot, 3, mtot.domains())
    mf2     = MeshFunction('size_t', mtot, 2, mtot.domains())
    m_in    = SubMesh(mtot,mf3,3)
    m_out   = SubMesh(mtot,mf3,1)

    mf2_out = MeshFunction('size_t', m_out, 2, m_out.domains())

    # time stepping
    dt      = 0.1
    Tfinal  = 0.5

    # problem on inner ball
    V_in    = FunctionSpace(m_in, "CG", 1)
    u_in    = project(Constant(5), V_in)
    un_in   = project(Constant(5), V_in)
    ub_in   = Function(V_in)
    v_in    = TestFunction(V_in)

    # problem on outer ball
    V_out   = FunctionSpace(m_out, "CG", 1)
    u_out   = Function(V_out)
    un_out  = Function(V_out)
    ub_out  = Function(V_out)
    v_out   = TestFunction(V_out)

    # projections
    LagrangeInterpolator.interpolate(ub_out, u_in)
    LagrangeInterpolator.interpolate(ub_in, u_out)

    # set extrapolation
    u_in.set_allow_extrapolation(True)
    u_out.set_allow_extrapolation(True)
    ub_in.set_allow_extrapolation(True)
    ub_out.set_allow_extrapolation(True)

    # measures
    dx_in   = Measure('dx', domain=m_in)
    ds_in   = Measure('ds', domain=m_in)
    dx_out  = Measure('dx', domain=m_out)
    ds_out  = Measure('ds', domain=m_out, subdomain_data=mf2_out)

    def w_in(n):
        return project((1+x[0]+x[1]+x[2])/n, V_in)
    def w_out(n):
        return project((1+x[0]+x[1]+x[2])/n, V_out)


    # functional forms
    Finner  = (u_in - un_in)/dt*v_in*dx_in     + inner(grad(u_in), grad(v_in))*dx_in    - (ub_in  - u_in  - w_in(n))*v_in*ds_in        
    Fouter  = (u_out - un_out)/dt*v_out*dx_out + inner(grad(u_out), grad(v_out))*dx_out - (ub_out - u_out + w_out(n))*v_out*ds_out(4) 

    # solver loop
    vtkin   = File("u_in.pvd")
    vtkout  = File("u_out.pvd")
    vtkin  << (u_in, 0)
    vtkout << (u_out, 0)

    t = dt
    idx = 1


    while t <= Tfinal:
        print(f"\n*****************\nSolving time step {idx}, (t={t})\n*****************\n")
        mass = assemble(u_in*dx_in) + assemble(u_out*dx_out)
        in_volume = assemble(Constant(1)*dx_in)
        out_volume = assemble(Constant(1)*dx_out)
        print(f"Total mass: {mass}")
        print(f"Inner volume: {in_volume}")
        print(f"Outer volume: {out_volume}")

        multiphys_loop([Fouter, Finner], [u_out, u_in], [ub_in, ub_out])


        un_out.assign(u_out)
        un_in.assign(u_in)

        vtkin  << (u_in, t)
        vtkout << (u_out, t)

        t   += dt
        idx += 1


    return u_in, u_out, w_in(n)


#nvec           = [1,3,10,30,1e2,3e2,1e3,3e3,1e4]
nvec = [10]
#nvec           = [1e2, 3e2, 1e3]
uin_vec        = []
uout_vec       = []
wn_vec         = []
uin_Linfvec    = []
uout_Linfvec   = []
utot_Linfvec   = []
w_Linfvec      = []


for idx, n in enumerate(nvec):
    uin, uout, wn = problem(n)
    uin_vec.append(uin)
    uout_vec.append(uout)
    wn_vec.append(wn)

    if idx == 0:
        continue
    uin_Linfvec.append(compute_Linf(uin_vec[idx], uin_vec[idx-1]))
    uout_Linfvec.append(compute_Linf(uout_vec[idx], uout_vec[idx-1]))
    utot_Linfvec.append(max(uin_Linfvec[-1], uout_Linfvec[-1]))
    w_Linfvec.append(compute_Linf(wn_vec[idx], wn_vec[idx-1]))



#x_equals_y = np.linspace(0,max(w_Linfvec),10000)
#
#
#uplot = utot_Linfvec
## normal 
#ax0 = plt.subplot(1,2,1)
#ax0.plot(w_Linfvec, uplot, 'o')
##ax0.plot(x_equals_y, x_equals_y, ':')
#ax0.set_xlabel('$||w^{(n+1)}-w^{(n)}||_{L_\infty}$')
#ax0.set_ylabel('$||u^{(n+1)}-u^{(n)}||_{L_\infty}$')
#slope, intercept, r_value, p_value, std_err = stats.linregress(w_Linfvec, uplot)
#text0 = f"slope = {slope:.4f}\nintercept = {intercept:.4f}\n$r^2$ = {(r_value**2):.4f}"
#ax0.text(0.2,0.8,text0,horizontalalignment='center', 
#         verticalalignment='center', transform=ax0.transAxes)
#
#
## log
#ax1 = plt.subplot(1,2,2)
#ax1.plot(np.log10(w_Linfvec), np.log10(uplot), 'o')
#ax1.set_xlabel('$log_{10} ||w^{(n+1)}-w^{(n)}||_{L_\infty}$')
#ax1.set_ylabel('$log_{10} ||u^{(n+1)}-u^{(n)}||_{L_\infty}$')
#slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(w_Linfvec), np.log10(uplot))
#text1 = f"slope = {slope:.4f}\nintercept = {intercept:.4f}\n$r^2$ = {(r_value**2):.4f}"
#ax1.text(0.2,0.8,text1,horizontalalignment='center', 
#         verticalalignment='center', transform=ax1.transAxes)
##ax1.plot(np.log10(x_equals_y), np.log10(x_equals_y), ':')
#
#plt.show()

