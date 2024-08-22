import numpy as np
import Mesh, fenics, sys, os
from scipy import linalg as sl 
from scipy import sparse as sp

# msh = Mesh.MeshData('32p', True)

def ObsMat(N):
    pN = int(np.sqrt(N))
    Obs = []
    Ufix = []
    k = -1
    for i in range(pN):
        for j in range(pN):
            k += 1
            if i == 0:
                Ufix.append(k)
                continue
            if j == 0 or j == pN-1 or i == pN-1:
                Obs.append(k)
                continue
    Mobs = sp.csr_matrix((len(Obs),N)).tolil()
    Mfix = sp.csr_matrix((len(Ufix),N)).tolil()
    for i, v in enumerate(Obs):
        Mobs[i, v] = 1.0
    for i, v in enumerate(Ufix):
        Mfix[i, v] = 1.0
    return Mobs, Mfix
        
def node2array(mshf):
    '''
    Returns:    A2N: 1D array   shape = (N)
                     the k element in the regular vector corresponding to the 
                     A2N[k] element in the fenics node vector
                N2A: 1D array   shape = (N)
                     the k element in the fenics node vector corresponding to the 
                     N2A[k] element in the regular vector
                Ma2n: sp.csr matrix shape = (N, N)
                     to convert a vector in fenics node order to a vector in regular 
                     order, use Ma2n.dot(v)
                Mn2a: sp.csr matrix shape = (N, N)
                     to convert a vector in regular order to a vector in fenics node  
                     order, use Mn2a.dot(v)
    '''
    U = fenics.FunctionSpace(mshf.mesh, 'CG', 1)
    u0 = fenics.interpolate(fenics.Expression('x[0]', degree=1), U)
    u1 = fenics.interpolate(fenics.Expression('x[1]', degree=1), U)
    uc0 = u0.vector().get_local()
    uc1 = u1.vector().get_local()
    N = uc0.shape[0]
    pN = int(np.sqrt(N))
    uc = np.zeros((N,2))
    uc[:,0], uc[:,1] = uc0, uc1

    dn = 1.0/(pN-1)
    A2N = np.empty(N, dtype=int)
    N2A = np.empty(N, dtype=int)
    k = 0
    for i in range(pN):
        for j in range(pN):
            vc0 = -0.5+dn*j
            vc1 = 0.5-dn*i
            for m in range(N):
                if abs(uc[m,0]-vc0)+abs(uc[m,1]-vc1) <= 1e-8:
                    A2N[k] = m
                    N2A[m] = k
                    break
            k += 1
    Ma2n, Mn2a = sp.csr_matrix((N,N)).tolil(), sp.csr_matrix((N,N)).tolil()
    for i in range(N):
        Ma2n[i, A2N[i]] = 1
        Mn2a[i, N2A[i]] = 1
    return A2N, N2A, Ma2n.tocsr(), Mn2a.tocsr()

'''
A2N, N2A, Ma2n, Mn2a = node2array(msh)
U = fenics.FunctionSpace(msh.mesh, 'CG', 1)
u0 = fenics.interpolate(fenics.Expression('x[0]', degree=1), U)
u1 = fenics.interpolate(fenics.Expression('x[1]', degree=1), U)
uc0 = u0.vector().get_local()
uc1 = u1.vector().get_local()
N = uc0.shape[0]
pN = int(np.sqrt(N))
uc = np.zeros((N,2))
uc[:,0], uc[:,1] = uc0, uc1
print(uc)
vc0 = Ma2n.dot(uc0)
vc1 = Ma2n.dot(uc1)
vc = np.zeros((N,2))
vc[:,0], vc[:,1] = vc0, vc1
print(vc)
print(vc0.reshape((pN,pN)))
print(vc1.reshape((pN,pN)))
'''

def full_order_solution_m(mesh_data, mul):
    '''
    solve the microscale full order solution of mu
    parallel version
    Parameters: mesh_data: MeshData()
                        the mesh, the cellmarkers and the facetmarkers
                mul: 2D array   shape = (n, N_para)
                    the parameters to be solve
    Returns:    u_matrix: 2D array   shape = (n, N_full_u)
                        the collection of solutions u
                p_matrix: 2D array   shape = (n, N_full_p)
                        the collection of flux p
    '''
    fenics.parameters["form_compiler"]["cpp_optimize"] = True
    fenics.parameters["form_compiler"]["quadrature_degree"] = 3
    ffc_options = {
        "optimize": True, "eliminate_zeros": True, \
        "precompute_basis_const": True, "precompute_ip_const": True
    }
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    N_processes = comm.Get_size()

    A2N, N2A, Ma2n, Mn2a = node2array(mesh_data)
    Mobs, Mfix = ObsMat(Ma2n.shape[0])
    # UUU = fenics.TensorFunctionSpace(mesh_data.mesh, 'DG', 0, shape=(2,2))
    U = fenics.VectorFunctionSpace(mesh_data.mesh, 'CG', 1, dim=2)
    dm0 = U.sub(0).dofmap()
    dm1 = U.sub(1).dofmap()
    U1 = fenics.FunctionSpace(mesh_data.mesh, 'CG', 1)
    ds_l = fenics.Measure("ds", domain=mesh_data.mesh, subdomain_data=mesh_data.facetmarkers, subdomain_id=3)
    ds_r = fenics.Measure("ds", domain=mesh_data.mesh, subdomain_data=mesh_data.facetmarkers, subdomain_id=5)
    ds_b = fenics.Measure("ds", domain=mesh_data.mesh, subdomain_data=mesh_data.facetmarkers, subdomain_id=6)
    u_ = fenics.TestFunction(U)
    du = fenics.TrialFunction(U)
    u = fenics.Function(U)

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
    bcd = fenics.DirichletBC(U, fenics.Expression(("0.0", "0.0"), degree=2), mesh_data.facetmarkers, 4)
    #T_b = fenics.Expression(("0.0", "1.0*(0.75-0.5*abs(x[0]))"), mpi_comm=fenics.MPI.comm_self, degree=2)
    T_b = fenics.Expression(("0.0", "0.45"), mpi_comm=fenics.MPI.comm_self, degree=2)
    T_r = T_R(0.45)
    T_l = T_L(0.45)
    Id = fenics.Identity(u.geometric_dimension())
    Fm = Id + fenics.grad(u)
    Cm = Fm.T*Fm
    Ic = fenics.tr(Cm)
    Jm = fenics.det(Fm)
    Ef = fenics.Function(U1)
    Ef.vector().set_local(10.0*np.ones(Ma2n.shape[0]))
    E = expr(Ef)
    nu = fenics.Constant(0.3)
    mu, lmbda = E/(2*(1 + nu)), E*nu/((1 + nu)*(1 - 2*nu))
    psi = (mu/2)*(Ic - 2) - mu*fenics.ln(Jm) + (lmbda/2)*(fenics.ln(Jm))**2
    Pi = psi*fenics.dx - fenics.dot(T_b, u)*ds_b - fenics.dot(T_r, u)*ds_r - fenics.dot(T_l, u)*ds_l
    F = fenics.derivative(Pi, u, u_)
    J = fenics.derivative(F, u, du)
    def solve(Ev):
        u.vector().set_local(np.zeros(u.vector().get_local().shape))
        Et = Ev * 95.0 + 5.0
        Ef.vector().set_local(Et)
        E.v = Ef
        fenics.solve(F == 0, u, bcd, J=J,form_compiler_parameters=ffc_options)
        #u.rename("displacement", "displacement")
        #file = fenics.File(fenics.MPI.comm_self, os.path.join(os.path.abspath('.'), 'fig_example', 'displacement_%d.pvd'%my_rank))
        #file << u
        ux = u.vector().vec()[dm0.dofs()]
        uy = u.vector().vec()[dm1.dofs()]
        uc = np.zeros((2, ux.shape[0]))
        uc[0, :], uc[1, :] = ux.copy(), uy.copy()
        return uc
    
    if my_rank == 0:
        def divide2chunk(n_task, n_process):
            chunksize = n_task // n_process + 1
            rest = n_task % n_process
            task_list = [chunksize]*rest
            task_list.extend([chunksize-1]*(n_process-rest))
            for i in range(1, n_process):
                task_list[i] += task_list[i-1]
            return task_list
        n_task = mul.shape[0]
        task_list = divide2chunk(n_task, N_processes)

        muc = mul[:task_list[0],...]
        print("Process %d distributes parameters..." %my_rank)
        sys.stdout.flush()
        for p in range(1, N_processes):
            data_send = mul[task_list[p-1]:task_list[p],...]
            comm.send(data_send, dest=p)
        print("Process %d done sending!" %my_rank)
        sys.stdout.flush()
    else:
        muc = comm.recv(source=0)
        print("Process %d received the parameters!" %my_rank)
        sys.stdout.flush()
    
    n_task_p = muc.shape[0]
    i = 0
    for Emur in muc:
        i += 1
        print("********** Process %d is getting the %d/%d solution! **********" % (my_rank, i, n_task_p))
        sys.stdout.flush()
        Emu = Mn2a.dot(Emur.squeeze().reshape(-1))
        uc = solve(Emu)
        ux = Ma2n.dot(uc[0,:])
        uy = Ma2n.dot(uc[1,:])
        obsx = Mobs.dot(ux)
        obsy = Mobs.dot(uy)
        Ns = int(np.sqrt(ux.shape[0]))
        ux = ux.reshape((Ns, Ns))
        uy = uy.reshape((Ns, Ns))
        u_i = np.stack((ux, uy), axis=0)
        obs_i = np.stack((obsx, obsy), axis=0)
        if i == 1:
            u_matrix = u_i[None,...]
            obs_matrix = obs_i[None,...]
        else:
            u_matrix = np.append(u_matrix, u_i[None,...], axis=0)
            obs_matrix = np.append(obs_matrix, obs_i[None,...], axis=0)
    print("Process %d finished the computation!" %my_rank)
    sys.stdout.flush()

    if my_rank == 0:
        for p in range(1, N_processes):
            u_matrix_p, obs_matrix_p = comm.recv(source=p)
            u_matrix = np.append(u_matrix, u_matrix_p, axis=0)
            obs_matrix = np.append(obs_matrix, obs_matrix_p, axis=0)
        print("Process %d done!" %my_rank)
        sys.stdout.flush()
        return u_matrix, obs_matrix
    else:
        comm.send((u_matrix, obs_matrix), dest=0)
        print("Process %d done!" %my_rank)
        sys.stdout.flush()

def get_loss_components(mshf):
    U = fenics.FunctionSpace(mshf.mesh, 'CG', 1)
    UD = fenics.FunctionSpace(mshf.mesh, 'DG', 0)
    UU = fenics.VectorFunctionSpace(mshf.mesh, 'DG', 0, dim=2)
    dm0 = UU.sub(0).dofmap()
    dm1 = UU.sub(1).dofmap()
    ds_l = fenics.Measure("ds", domain=mshf.mesh, subdomain_data=mshf.facetmarkers, subdomain_id=3)
    ds_r = fenics.Measure("ds", domain=mshf.mesh, subdomain_data=mshf.facetmarkers, subdomain_id=5)
    ds_b = fenics.Measure("ds", domain=mshf.mesh, subdomain_data=mshf.facetmarkers, subdomain_id=6)
    A2N, N2A, Ma2n, Mn2a = node2array(mshf)
    N = Ma2n.shape[0]
    Id = np.identity(N)
    Idnode = Mn2a.dot(Id)
    u = fenics.Function(U)
    for i in range(N):
        vi = Idnode[:,i]
        u.vector().set_local(vi)

        vd = fenics.interpolate(u, UD)
        ud = vd.vector().get_local()

        gu = fenics.project(fenics.grad(u), UU)
        gux = gu.vector().vec()[dm0.dofs()]
        guy = gu.vector().vec()[dm1.dofs()]
        if i == 0:
            CG2DG = ud[..., None]
            Gradx = gux[..., None]
            Grady = guy[..., None]
        else:
            CG2DG = np.append(CG2DG, ud[..., None], axis=1)
            Gradx = np.append(Gradx, gux[..., None], axis=1)
            Grady = np.append(Grady, guy[..., None], axis=1)
    u_ = fenics.TestFunction(UD)
    itg = u_*fenics.dx
    Vitg = fenics.assemble(itg).get_local()

    class T_R(fenics.UserExpression):
        def __init__(self, val):
            super().__init__()
            self.val = val
        def eval(self, values, x):
            if x[1] > 0:
                values[0] = -self.val+2*self.val*x[1]
            else:
                values[0] = -self.val#-(0.75-0.5*abs(x[1]))
        def value_shape(self):
            return ()
    class T_L(fenics.UserExpression):
        def __init__(self, val):
            super().__init__()
            self.val = val
        def eval(self, values, x):
            if x[1] > 0:
                values[0] = self.val-2*self.val*x[1]
            else:
                values[0] = self.val#(0.75-0.5*abs(x[1]))
        def value_shape(self):
            return ()
    T_b = fenics.Expression(("0.45"), mpi_comm=fenics.MPI.comm_self, degree=2)
    T_r = T_R(0.45)
    T_l = T_L(0.45)
    v_ = fenics.TestFunction(U)
    s_y = - T_b*v_*ds_b
    s_x = - T_r*v_*ds_r - T_l*v_*ds_l
    S_Y = fenics.assemble(s_y).get_local()
    S_X = fenics.assemble(s_x).get_local()
    S_Y = Ma2n.dot(S_Y)
    S_X = Ma2n.dot(S_X)
    return CG2DG, Gradx, Grady, Vitg, S_X, S_Y