import numpy as np
import os, pickle, Samp, sys
import Mesh
import FEdep
from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
N_processes = comm.Get_size()

S = 2**16
RN = 32
Channel = 1
TS = 300

# lat_var = Samp.generate_latent_var(S, 1, 2)
# sam = Samp.generate_para_sample(RN, lat_var)
# lat_var = Samp.generate_latent_var(TS, 3, 4)
# Tsam = Samp.generate_para_sample(RN, lat_var)

if not os.path.exists(os.path.join(os.path.abspath('.'), 'KnownData')):
    os.mkdir(os.path.join(os.path.abspath('.'), 'KnownData'))
# with open(os.path.join(os.path.abspath('.'), 'KnownData', 'ParaSam' + '.pickle'), 'wb') as file:
#     pickle.dump((sam, Tsam), file)
with open(os.path.join(os.path.abspath('.'), 'KnownData', 'ParaSam' + '.pickle'), 'rb') as file:
    sam, Tsam = pickle.load(file)

'''   
msh = Mesh.MeshData('32p', True)
if my_rank == 0:
    u_sam, obs_sam = FEdep.full_order_solution_m(msh, sam[0:3000,...])
    with open(os.path.join(os.path.abspath('.'), 'KnownData', 'USam' + '.pickle'), 'wb') as file:
        pickle.dump((u_sam, obs_sam), file)
else:
    FEdep.full_order_solution_m(msh, sam[0:3000,...])

if my_rank == 0:
    u_Tsam, obs_Tsam = FEdep.full_order_solution_m(msh, Tsam)
    with open(os.path.join(os.path.abspath('.'), 'KnownData', 'UTSam' + '.pickle'), 'wb') as file:
        pickle.dump((u_Tsam, obs_Tsam), file)
else:
    FEdep.full_order_solution_m(msh, Tsam)
''' 
'''
msh = Mesh.MeshData('32p', True)
if my_rank == 0:
    CG2DG, Gradx, Grady, Vitg, S_X, S_Y = FEdep.get_loss_components(msh)
    Mobs, Mfix = FEdep.ObsMat(1024)
    A2N, N2A, Ma2n, Mn2a = FEdep.node2array(msh)
    with open(os.path.join(os.path.abspath('.'), 'KnownData', 'Losscomp' + '.pickle'), 'wb') as file:
        pickle.dump((A2N, N2A, Ma2n, Mn2a, Mobs, Mfix, CG2DG, Gradx, Grady, Vitg, S_X, S_Y), file)
'''
with open(os.path.join(os.path.abspath('.'), 'KnownData', 'Losscomp' + '.pickle'), 'rb') as file:
    A2N, N2A, Ma2n, Mn2a, Mobs, Mfix, CG2DG, Gradx, Grady, Vitg, S_X, S_Y = pickle.load(file)

import fenics
msh = Mesh.MeshData('32p', True)
U = fenics.VectorFunctionSpace(msh.mesh, 'CG', 1, dim=2)
dm0 = U.sub(0).dofmap()
dm1 = U.sub(1).dofmap()
s = sam[0, ...]
Emu = Mn2a.dot(s.squeeze().reshape(-1))
Ef = fenics.Function(U)
Ef.vector().vec()[dm0.dofs()] = Emu * 95.0 + 5.0
Ef.vector().vec()[dm1.dofs()] = 0.3 - Emu * 0.2
#Ef.rename("Young's modulus", "porousity radius")
file = fenics.File(fenics.MPI.comm_self, os.path.join(os.path.abspath('.'), 'fig_example', 'parameter_field.pvd'))
file << Ef

'''
# test
with open(os.path.join(os.path.abspath('.'), 'KnownData', 'USam' + '.pickle'), 'rb') as file:
    u_sam, obs_sam = pickle.load(file)
ux = u_sam[112,0,...].reshape(-1)
uy = u_sam[112,1,...].reshape(-1)
E = sam[112,0,...].reshape(-1)
E = E*95.0 + 5.0

import fenics
U = fenics.FunctionSpace(msh.mesh, 'CG', 1)
UD = fenics.FunctionSpace(msh.mesh, 'DG', 0)
UU = fenics.VectorFunctionSpace(msh.mesh, 'DG', 0, dim=2)
Ut = fenics.VectorFunctionSpace(msh.mesh, 'CG', 1, dim=2)
dm0 = UU.sub(0).dofmap()
dm1 = UU.sub(1).dofmap()
dmt0 = Ut.sub(0).dofmap()
dmt1 = Ut.sub(1).dofmap()
ds_l = fenics.Measure("ds", domain=msh.mesh, subdomain_data=msh.facetmarkers, subdomain_id=3)
ds_r = fenics.Measure("ds", domain=msh.mesh, subdomain_data=msh.facetmarkers, subdomain_id=5)
ds_b = fenics.Measure("ds", domain=msh.mesh, subdomain_data=msh.facetmarkers, subdomain_id=6)
u = fenics.Function(U)
u.vector().set_local(Mn2a.dot(ux))
v = fenics.Function(U)
v.vector().set_local(Mn2a.dot(uy))
Ef = fenics.Function(U)
Ef.vector().set_local(Mn2a.dot(E))

guxm = Gradx.dot(ux)
guym = Grady.dot(ux)
gvxm = Gradx.dot(uy)
gvym = Grady.dot(uy)
Em = CG2DG.dot(E)
psi = np.zeros(Em.shape[0])
for i in range(guxm.shape[0]):
    J = (1+guxm[i])*(1+gvym[i]) - guym[i]*gvxm[i]
    Ic = (1+guxm[i])**2 + (1+gvym[i])**2 + guym[i]**2 + gvxm[i]**2
    nu = 0.3
    mu, lmbda = Em[i]/(2*(1 + nu)), Em[i]*nu/((1 + nu)*(1 - 2*nu))
    psi[i] = (mu/2)*(Ic - 2) - mu*np.log(J) + (lmbda/2)*(np.log(J))**2
Psi = np.dot(Vitg, psi) + np.dot(S_X, ux) + np.dot(S_Y, uy)

class expr(fenics.UserExpression):
    def __init__(self, vec, **kwargs):
        super().__init__(**kwargs)
        self.v = vec
    def eval(self, values, x):
        values[0] = self.v(x)
    def value_shape(self):
        return ()
class T_R(fenics.UserExpression):
    def __init__(self, val):
        super().__init__()
        self.val = val
    def eval(self, values, x):
        if x[1] > 0:
            values[0] = -self.val+2*self.val*x[1]
            values[1] = 0
        else:
            values[0] = -self.val#-(0.75-0.5*abs(x[1]))
            values[1] = 0
    def value_shape(self):
        return (2,)
class T_L(fenics.UserExpression):
    def __init__(self, val):
        super().__init__()
        self.val = val
    def eval(self, values, x):
        if x[1] > 0:
            values[0] = self.val-2*self.val*x[1]
            values[1] = 0
        else:
            values[0] = self.val#(0.75-0.5*abs(x[1]))
            values[1] = 0
    def value_shape(self):
        return (2,)
T_b = fenics.Expression(("0.0", "0.45"), mpi_comm=fenics.MPI.comm_self, degree=2)
T_r = T_R(0.45)
T_l = T_L(0.45)
ut = fenics.Function(Ut)
ut.vector().vec()[dmt0.dofs()] = Mn2a.dot(ux)
ut.vector().vec()[dmt1.dofs()] = Mn2a.dot(uy)
Id = fenics.Identity(2)
Fm = Id + fenics.grad(ut)
Cm = Fm.T*Fm
Ic = fenics.tr(Cm)
Jm = fenics.det(Fm)
E1 = expr(Ef)
nu = fenics.Constant(0.3)
mu, lmbda = E1/(2*(1 + nu)), E1*nu/((1 + nu)*(1 - 2*nu))
psif = (mu/2)*(Ic - 2) - mu*fenics.ln(Jm) + (lmbda/2)*(fenics.ln(Jm))**2
Pi = psif*fenics.dx - fenics.dot(T_r, ut)*ds_r - fenics.dot(T_l, ut)*ds_l - fenics.dot(T_b, ut)*ds_b
Psi1 = fenics.assemble(Pi)
print(Psi)
print(Psi1)
psif = fenics.project(psif, UD)
'''