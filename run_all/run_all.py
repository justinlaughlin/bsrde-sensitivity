"""
Run all simulations:
 * elliptic bulk
 * parabolic bulk
 * elliptic surface
 * parabolic surface
 * parabolic bulk-surface
 * parabolic bulk-bulk
 * elliptic nonlinear

"""

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import ufl


#=========================================
# General Definitions
#=========================================
nvec = [1,3,10,30,1e2,3e2,1e3,3e3,1e4]

# perturbance function
def w_null(n, V, x, deg=None):
    return project(0, V)

def w_n(n, V, x, deg=1):
    return project((1+x[0]+x[1])**deg/n, V)

# perturbance function for dirichlet bc
def g_n(n):
    return Expression(f"(1+x[0]+x[1])/{n}", degree=1)

# compute Linf norm of absolute difference in solutions
def compute_Linf(u_0, u_1):
    return max(abs(u_0.compute_vertex_values() - u_1.compute_vertex_values()))

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

#=========================================
# Problems
#=========================================

"""
PDE types:
1) Linear Elliptic (scalar)
2) Linear Elliptic System 
3) Nonlinear Elliptic (scalar) (w/ RHS nonlinearity)
4) Nonlinear Elliptic System (w/ RHS nonlinearity)
5) Linear Parabolic (scalar)
7) Linear Parabolic System
6) Nonlinear Parabolic (scalar)
8) Nonlinear Parabolic System 

Geometries:
i)   2-dim region in R2                         : disc                      (scalar or vector)
ii)  2-dim manifold embedded in R3              : sphere                    (scalar or vector)
iii) 3-dim region in R3                         : ball                      (scalar or vector)
iv)  3-dim manifold coupled to 2-dim manifold   : ball coupled to sphere    (vector)
v)   3-dim manifold coupled to 3-dim manifold   : inner-outer balls         (vector)

Perturbations:
a) add to RHS of equation
b) in RHS of equation (for nonlinear)
c) add to boundary condition (neumann)
d) in boundary condition (neumann)
e) add to boundary condition (dirichlet)
f) in boundary condition (dirichlet)
g) in initial condition

Perturbation types:
* 1st order poly

======================
Cases done so far
======================
1.ii.a
2.ii.a

======================
Possible PDE/Geometry combinations
======================
1: i, ii, iii
2: 

"""




def get_geometry(gcase=1):
    """
    Geometries:
    i)   R2 manifold in R2                      : disc
    ii)  R2 manifold embedded in R3             : sphere
    iii) R3 manifold in R3                      : ball
    iv)  R3 manifold coupled to R2 manifold     : ball coupled to sphere
    v)   R3 manifold coupled to R3 manifold     : inner-outer balls
    """
    if gcase==1:
        N       = 50
        comm    = MPI.comm_world
        m       = UnitDiscMesh.create(comm, N, 1, 3)
        mb      = None
    if gcase==2:
        temp   = Mesh("unit_ball.xml")
        m       = BoundaryMesh(temp, "exterior")
        mb      = None
    if gcase==3:
        m       = Mesh("unit_ball.xml")
        mb      = None
    if gcase==4:
        m       = Mesh("unit_ball.xml")
        mb      = BoundaryMesh(m, "exterior")
    if gcase==5:
        m       = Mesh("inner_outer_balls.xml")
        x       = MeshCoordinates(m)
        mf3     = MeshFunction('size_t', m, 3, m.domains())
        mf2     = MeshFunction('size_t', m, 2, m.domains())
        m_in    = SubMesh(m,mf3,3)
        m_out   = SubMesh(m,mf3,1)

        return m, m_in, m_out

    return m, mb

# def ball_sphere_geometry():

#     m = Mesh("unit_ball.xml")
#     mb = BoundaryMesh(m, "exterior")
#     x = MeshCoordinates(m)
#     dx = Measure("dx", m)
#     dxb = Measure("dx", mb) 

#     V = FunctionSpace(m,"CG",1)
#     Vp = VectorFunctionSpace(m,"CG",1,dim=2) # Vb projected onto m
    
#     Vb = VectorFunctionSpace(mb,"CG",1,dim=2)
#     Vpb = FunctionSpace(mb,"CG",1) # V projected onto mb


def problem_elliptic_scalar(n, gcase=1, rhs=None, w_rhs=None, w_g=None, w_h=None, wdeg=1):
    m, mb   = get_geometry(gcase=gcase)
    V       = FunctionSpace(m,"CG",1)
    x       = MeshCoordinates(m)
    dx      = Measure("dx", m)
    def boundary(x, on_boundary):
        return x[0] > 0.6

    u       = Function(V)
    v       = TestFunction(V)
    bc      = DirichletBC(V, Constant(0), boundary)

    w = w_rhs(n,V,x,wdeg)
    
    F = inner(grad(u), grad(v))*dx - (rhs(u) + w)*v*dx
    solve(F==0,u,bc)
    #F = inner(grad(u), grad(v))*dx - (-j*v[0] + -j*v[1] + j*v[2] + w_n(n)*(v[0]+v[1]+v[2]))*dx

    return u, w


def nonlinearity_sinh(u):
    return ufl.sinh(u)

problem_elliptic_scalar(5, rhs=nonlinearity_sinh, w_rhs=w_n, wdeg=1)


def get_problem_slope(wvec, problem, gcase=1, rhs=None, w_rhs=None, w_g=None, w_h=None, wdeg=1):
    uvec = []
    wvec = []
    uLinfvec = []
    wLinfvec = []
    for n in nvec:
        u, w = problem(n, gcase, rhs, w_rhs, w_g, w_h, wdeg)
        uvec.append(u)
        wvec.append(w)

        if idx == 0:
            continue
        uLinfvec.append(compute_Linf(uvec[idx], uvec[idx-1]))
        wLinfvec.append(compute_Linf(w_n(nvec[idx]), w_n(nvec[idx-1])))

# def problem_parabolic_coupled(m):

#     u = Function(V)
#     up = Function(Vp)
#     u_ = project(Constant(0), V)

#     ub = Function(Vb)
#     upb = Function(Vpb)
#     ub_ = project(as_vector([Constant(0), Constant(0)]), Vb)
#     wp_n = Function(Vpb)

#     LagrangeInterpolator.interpolate(up, ub)
#     LagrangeInterpolator.interpolate(upb, u)
#     LagrangeInterpolator.interpolate(wp_n, w_n(n))

#     v = TestFunction(V)
#     vb = TestFunction(Vb)

#     dt = 0.01
#     t = dt
#     bc = []
    
#     jf = k0*u*ub[0]
#     jr = k1*ub[1]
#     j = jf - jr


#     #j = k0*u*up[0] - k1*up[1]
#     #jb = k0*upb*ub[0] - k1*ub[1]
#     #F0 = inner((u-u_)/dt, v)*dx + inner(grad(u), grad(v))*dx - (- j + w_n(n))*v*ds
#     #F1 = inner((ub-ub_)/dt, vb)*dxb + inner(grad(ub), grad(vb))*dxb - (-jb*vb[0] + jb*vb[1] + wp_n*(vb[0]+vb[1]))*dxb

#     j = k0*u*(up[0]+w_n(n)) - k1*(up[1]+w_n(n))
#     jb = k0*upb*(ub[0]+wp_n) - k1*(ub[1]+wp_n)
#     #j = k0*u*(sinh+w_n(n)) - k1*(up[1]+w_n(n))
#     #jb = k0*upb*(ub[0]+wp_n) - k1*(ub[1]+wp_n)
#     F0 = inner((u-u_)/dt, v)*dx + inner(grad(u), grad(v))*dx - (- j + w_n(n))*v*ds
#     F1 = inner((ub-ub_)/dt, vb)*dxb + inner(grad(ub), grad(vb))*dxb - (-jb*vb[0] + jb*vb[1] + wp_n*(vb[0]+vb[1]))*dxb

#     vtkfile = File(f'u_n={n}_.pvd')
#     vtkfileb = File(f'ub_n={n}_.pvd')

#     while t < 0.1:
#         solve(F0==0, u, bc)
#         solve(F1==0, ub, bc)

#         LagrangeInterpolator.interpolate(up, ub)
#         LagrangeInterpolator.interpolate(upb, u)
#         LagrangeInterpolator.interpolate(wp_n, w_n(n))

#         t += dt
#         u_.assign(u)
#         ub_.assign(ub)
#         #vtkfile << (u,t)
#         #vtkfileb << (ub,t)

#     return u, ub

# def Problem_7(n):
#     comm = MPI.comm_world
#     m = UnitDiscMesh.create(comm, 50, 1, 3)

#     x = MeshCoordinates(m)
#     dx = Measure("dx", m)
#     class Boundary(SubDomain):
#         def inside(self, x, on_boundary):
#             return x[0] > (xmax-DOLFIN_EPS)

#     def boundary(x, on_boundary):
#         return on_boundary # x[0] > 0.99

#     # functions
#     V = FunctionSpace(m,"CG",1)

#     # problem
#     u = Function(V)
#     v = TestFunction(V)
#     #g = lambda n: Expression(f"(1+x[0]+x[1])/{n}", degree=1)
#     #bc = DirichletBC(V, Constant(0.0), boundary)
#     bc = DirichletBC(V, g_n(n), boundary)
    
#     F = inner(grad(u), grad(v))*dx + ufl.sinh(u)*v*dx

#     solve(F==0, u, bc)
#     #File(f'u_n={n}.pvd') << u

