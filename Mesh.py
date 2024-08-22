import fenics, meshio, os, sys

def msh_read(msh_str, prune_z=False):
    """
    Input the name of mesh file in .msh format (without extension)
    Return three object: Mesh object, CellFunctionSizet object and FacetFunctionSizet object
    The later two objects contain the subdomain information
    """
    msh_str_msh = msh_str + ".msh"    
    msh = meshio.read(msh_str_msh)

    for key in msh.cells_dict.keys():
        if key == "line":
            line_cells = msh.cells_dict[key]
        elif key == "triangle":
            triangle_cells = msh.cells_dict[key]
        elif key == "tetra":
            tetra_cells = msh.cells_dict[key]

    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "line":
            line_data = msh.cell_data_dict["gmsh:physical"][key]
        elif key == "triangle":
            triangle_data = msh.cell_data_dict["gmsh:physical"][key]
        elif key == "tetra":
            tetra_data = msh.cell_data_dict["gmsh:physical"][key]
    '''
    tetra_mesh = meshio.Mesh(points = msh.points, 
                                cells = [("tetra", tetra_cells)],
                                cell_data = {"name_to_read":[tetra_data]})
                            '''
    
    triangle_mesh = meshio.Mesh(points = msh.points, 
                                    cells = [("triangle", triangle_cells)], 
                                    cell_data = {"name_to_read":[triangle_data]})
    line_mesh =meshio.Mesh(points = msh.points,
                            cells = [("line", line_cells)],
                            cell_data = {"name_to_read":[line_data]})
    if prune_z: 
        triangle_mesh.prune_z_0()
        line_mesh.prune_z_0()
    msh_str_cell = os.path.join(os.path.abspath('.'), 'mesh_data', msh_str + '_cell.xdmf')
    msh_str_facet = os.path.join(os.path.abspath('.'), 'mesh_data', msh_str + '_facet.xdmf')
    if not os.path.exists(os.path.join(os.path.abspath('.'), 'mesh_data')):
        os.mkdir(os.path.join(os.path.abspath('.'), 'mesh_data'))
    
    #meshio.write(msh_str_cell, triangle_mesh)
    #meshio.write(msh_str_facet, line_mesh)
    
    mesh = fenics.Mesh(fenics.MPI.comm_self)
    with fenics.XDMFFile(fenics.MPI.comm_self, msh_str_cell) as infile:
        infile.read(mesh)
        mvc = fenics.MeshValueCollection("size_t", mesh, 2)
        infile.read(mvc, "name_to_read")
    cell_markers = fenics.cpp.mesh.MeshFunctionSizet(mesh, mvc)
    with fenics.XDMFFile(fenics.MPI.comm_self, msh_str_facet) as infile:
        mvc = fenics.MeshValueCollection("size_t", mesh, 1)
        infile.read(mvc, "name_to_read")
    facet_markers = fenics.cpp.mesh.MeshFunctionSizet(mesh, mvc)
    
    return mesh, cell_markers, facet_markers

class MeshData():
    mesh = None # main mesh
    cellmarkers = None  # mark the subdomain
    facetmarkers = None # mark the boundary and interface
    submesh_M = None    # submesh for the matrix
    submesh_I = None    # submesh for the inclusion
    def __init__(self, msh_str, prune_z=False):
        self.mesh, self.cellmarkers, self.facetmarkers = msh_read(msh_str, prune_z)
        #self.submesh_M = fenics.SubMesh(self.mesh, self.cellmarkers, 0) # submesh of matrix dx(0, domain=mh, subdomain_data=cm)
        #self.submesh_I = fenics.SubMesh(self.mesh, self.cellmarkers, 1) # submesh of inclusion dx(1, domain=mh, subdomain_data=cm)
    
    def get_boundary_nodes(self):
        l_index, t_index, r_index, b_index, i_index = set(), set(), set(), set(), set()
        for f in fenics.facets(self.mesh):
            if self.facetmarkers.array()[f.index()] == 3: 
                for v in fenics.vertices(f): 
                    l_index.add(v.index())
            elif self.facetmarkers.array()[f.index()] == 4: 
                for v in fenics.vertices(f): 
                    t_index.add(v.index())
            elif self.facetmarkers.array()[f.index()] == 5: 
                for v in fenics.vertices(f): 
                    r_index.add(v.index())
            elif self.facetmarkers.array()[f.index()] == 6: 
                for v in fenics.vertices(f): 
                    b_index.add(v.index())
            elif self.facetmarkers.array()[f.index()] == 2: 
                for v in fenics.vertices(f): 
                    i_index.add(v.index())
        return list(l_index), list(t_index), list(r_index), list(b_index), list(i_index)
