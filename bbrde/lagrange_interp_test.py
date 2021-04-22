from dolfin import *

# mesh
mtot    = Mesh("inner_outer_balls.xml")

mf3     = MeshFunction('size_t', mtot, 3, mtot.domains())
m_in    = SubMesh(mtot,mf3,3)
m_out   = SubMesh(mtot,mf3,1)

mf2_out = MeshFunction('size_t', m_out, 2, m_out.domains())

x       = MeshCoordinates(mtot)
x_out   = MeshCoordinates(m_out)
x_in    = MeshCoordinates(m_in)


# define function on inner ball
V_in    = FunctionSpace(m_in, "CG", 1)
V_out   = FunctionSpace(m_out, "CG", 1)

f       = project(20*((x_in[0]**2 + x_in[1]**2 + x_in[2]**2)**0.5), V_in)
u_out   = Function(V_out)

LagrangeInterpolator.interpolate(u_out, f)


File("f.pvd") << f
File("u.pvd") << u_out
