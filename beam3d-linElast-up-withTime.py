from fenics import *

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['quadrature_degree'] = 3

list_linear_solver_methods()

def strain(u):
    return sym(grad(u))

# time constants
t = 0.0; dt = 0.2; Tfinal = 20.0


#  ********* Create domain and mesh  ******* #

nps = 4

mesh = BoxMesh(Point(0, 0, 0), Point(10, 1, 1), nps*10, nps, nps)
nn = FacetNormal(mesh)
fileO = File(mesh.mpi_comm(), "out/Beam-Locking-linear.pvd")
# fileO.parameters["functions_share_mesh"] = True

#  ********* Defintion of function spaces  ******* #

P2v = VectorElement("CG", mesh.ufl_cell(), 2)
P1dc = FiniteElement("DG", mesh.ufl_cell(), 1)
Vh  = FunctionSpace(mesh, P2v)
Hh = FunctionSpace(mesh, MixedElement([P2v,P1dc]))


print(" **************** P2-P1disc Dof = ", Hh.dim())

# ******* Mechanic parameters ******* #
E     = Constant(1.0)
nu    = Constant(0.4)
lmbda = E*nu/((1.0+nu)*(1.0-2.0*nu))
mu    = E/(2.0*(1.0+nu))
rho   = Constant(10.0)

print("lmbda = %g, mu = %g" % (float(lmbda), float(mu)))

# ********* Boundary  conditions, forcing terms ******* #

bdry = MeshFunction("size_t", mesh, 2) 
bdry.set_all(0)

Clamp = CompiledSubDomain("near(x[0],0) && on_boundary")
Clamp.mark(bdry, 71)

Press = CompiledSubDomain("near(x[2],0) && on_boundary")
Press.mark(bdry, 72)

ds = Measure("ds", subdomain_data=bdry)


bcs = DirichletBC(Hh.sub(0), Constant((0, 0, 0)), bdry, 71)
ff = Expression(("0.0","0.0","-9.81"),degree=3)
# +(1-x[2] + 0.2*(x[2]-0.5)*x[0])

# *************  Mixed variational form ************** #

up = TrialFunction(Hh)
vq = TestFunction(Hh)

(u,p) = split(up)
(v,q) = split(vq)

uold = Function(Vh)
uuold = Function(Vh)

MLeft = rho*1.0 / pow(dt,2.0) * dot(u,v) * dx \
        + 2*mu*inner(strain(u),strain(v))*dx - p*div(v)*dx \
        - (p/lmbda+div(u))*q*dx

MRight = rho*dot(ff,v)*dx \
        + rho*2.0 / pow(dt,2.0) * dot(uold,v) * dx \
        - rho*1.0 / pow(dt,2.0) * dot(uuold,v) * dx

# ********* Solving the problem ******** #

sol = Function(Hh)

# ********* Time loop ************* #
while (t <= Tfinal):

    print("t=%.2f" % t)

    # updating the time-dependent traction
    pN.t = t


    solve(MLeft == MRight, sol, bcs, solver_parameters={'linear_solver':'lu'})

    u,p = sol.split()
    assign(uuold,uold); assign(uold,u) 

    u.rename("u","u"); p.rename("p","p")
    fileO.write(u,t); fileO.write(p,t);
    t += dt; 
