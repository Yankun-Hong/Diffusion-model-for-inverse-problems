import fenics

class K_scalar(fenics.UserExpression):
    def __init__(self, materials, k_0=1.0, k_1=1.0, **kwargs):
        super().__init__(**kwargs)
        self.materials = materials
        self.k_0 = k_0
        self.k_1 = k_1
    def eval_cell(self, values, x, cell):
        if self.materials[cell.index] == 0:
            values[0] = self.k_0
        elif self.materials[cell.index] == 1: 
            values[0] = self.k_1
    def value_shape(self):
        """
        Decide the vectorial expression's dimension, i.e. rank.
        Set (n,) for n-dim vector. If scalar, should be set as ().
        This value should be consistent with the VectorFunctionSpace dim. 
        """
        return () 

class K_vector(fenics.UserExpression):
    def __init__(self, materials, k_0=[1.0,0.0], k_1=[1.0,0.0], **kwargs):
        """
        only for 2D vector, inputs should be a list
        """
        super().__init__(**kwargs)
        self.materials = materials
        self.k_0 = k_0
        self.k_1 = k_1
    def eval_cell(self, values, x, cell):
        if self.materials[cell.index] == 0:
            values[0] = self.k_0[0]
            values[1] = self.k_0[1]
        elif self.materials[cell.index] == 1: 
            values[0] = self.k_1[0]
            values[1] = self.k_1[1]
    def value_shape(self):
        """
        Decide the vectorial expression's dimension, i.e. rank.
        Set (n,) for n-dim vector. If scalar, should be set as ().
        This value should be consistent with the VectorFunctionSpace dim. 
        """
        return (2,) 
    def split(self, **kwargs):
        """
        split the tensor into its components(K_scalar)
        """
        k0 = K_scalar(self.materials, k_0=self.k_0[0], k_1=self.k_1[0], **kwargs)
        k1 = K_scalar(self.materials, k_0=self.k_0[1], k_1=self.k_1[1], **kwargs)
        return k0, k1

class K_tensor(fenics.UserExpression):
    def __init__(self, materials, k_0=[[1.0,0.0],[0.0,1.0]], k_1=[[1.0,0.0],[0.0,1.0]], **kwargs):
        """
        only for 2 order tensor, inputs should be 2D list
        """
        super().__init__(**kwargs)
        self.materials = materials
        self.k_0 = k_0
        self.k_1 = k_1
    def eval_cell(self, values, x, cell):
        if self.materials[cell.index] == 0:
            values[0] = self.k_0[0][0]
            values[1] = self.k_0[0][1]
            values[2] = self.k_0[1][0]
            values[3] = self.k_0[1][1]
        elif self.materials[cell.index] == 1: 
            values[0] = self.k_1[0][0]
            values[1] = self.k_1[0][1]
            values[2] = self.k_1[1][0]
            values[3] = self.k_1[1][1]
    def value_shape(self):
        """
        Decide the vectorial expression's dimension, i.e. rank.
        Set (n,) for n-dim vector. If scalar, should be set as ().
        This value should be consistent with the VectorFunctionSpace dim. 
        """
        return (2,2) 
    def split(self, **kwargs):
        """
        split the tensor into its components(K_scalar)
        """
        k00 = K_scalar(self.materials, k_0=self.k_0[0][0], k_1=self.k_1[0][0], **kwargs)
        k01 = K_scalar(self.materials, k_0=self.k_0[0][1], k_1=self.k_1[0][1], **kwargs)
        k10 = K_scalar(self.materials, k_0=self.k_0[1][0], k_1=self.k_1[1][0], **kwargs)
        k11 = K_scalar(self.materials, k_0=self.k_0[1][1], k_1=self.k_1[1][1], **kwargs)
        return k00, k01, k10, k11
    def split_col(self, **kwargs):
        """
        Split the tensor into two columns (K_vector)
        """
        k0 = K_vector(self.materials, k_0=[self.k_0[0][0],self.k_0[1][0]], k_1=[self.k_1[0][0],self.k_1[1][0]], **kwargs)
        k1 = K_vector(self.materials, k_0=[self.k_0[0][1],self.k_0[1][1]], k_1=[self.k_1[0][1],self.k_1[1][1]], **kwargs)
        return k0, k1
    def split_row(self, **kwargs):
        """
        Split the tensor into two columns (K_vector)
        """
        k0 = K_vector(self.materials, k_0=self.k_0[0], k_1=self.k_1[0], **kwargs)
        k1 = K_vector(self.materials, k_0=self.k_0[1], k_1=self.k_1[1], **kwargs)
        return k0, k1

class Transf_vector(fenics.UserExpression):
    def __init__(self, r, **Kwargs) -> None:
        super().__init__(**Kwargs) 
        self.rm = 0.45
        self.r0 = 0.2
        self.r = r 
    def eval(self, values, x):
        xn = (x[0]**2 + x[1]**2)**0.5
        if xn < self.rm and xn > self.r0-1e-6:
            sc = (self.rm*(self.r-self.r0))/(xn*(self.rm-self.r0)) + (self.rm-self.r)/(self.rm-self.r0)
            values[0] = sc*x[0]
            values[1] = sc*x[1]
        else:
            values[0] = x[0]
            values[1] = x[1]
    def value_shape(self):
        return (2,)

class detJJinv(fenics.UserExpression):
    def __init__(self, r, **Kwargs) -> None:
        super().__init__(**Kwargs) 
        self.rm = 0.45
        self.r0 = 0.2
        self.r = r 
        self.rdif = self.rm - self.r0
    def eval(self, values, x):
        xn = (x[0]**2 + x[1]**2)**0.5
        if xn < self.rm and xn > self.r0-1e-6:
            values[0] = self.r*(x[0]*x[0]*self.rm/(xn**3)-1)/self.rdif - self.rm*(x[0]*x[0]*self.r0/(xn**3)-1)/self.rdif
            values[1] = self.r*(x[0]*x[1]*self.rm/(xn**3))/self.rdif - self.rm*(x[0]*x[1]*self.r0/(xn**3))/self.rdif
            values[2] = self.r*(x[1]*x[0]*self.rm/(xn**3))/self.rdif - self.rm*(x[1]*x[0]*self.r0/(xn**3))/self.rdif
            values[3] = self.r*(x[1]*x[1]*self.rm/(xn**3)-1)/self.rdif - self.rm*(x[1]*x[1]*self.r0/(xn**3)-1)/self.rdif
        else:
            values[0] = 1.0
            values[1] = 0.0
            values[2] = 0.0
            values[3] = 1.0
    def value_shape(self):
        return (2,2)

class detJ(fenics.UserExpression):
    def __init__(self, r, **Kwargs) -> None:
        super().__init__(**Kwargs) 
        self.rm = 0.45
        self.r0 = 0.2
        self.r = r 
        self.rdif = self.rm - self.r0
    def eval(self, values, x):
        xn = (x[0]**2 + x[1]**2)**0.5
        if xn < self.rm and xn > self.r0-1e-6:
            values[0] = (self.r)**2*(1-self.rm/xn)/((self.rdif)**2) + self.r*self.rm*((self.r0+self.rm)/xn-2)/((self.rdif)**2) + (self.rm)**2*(1-self.r0/xn)/((self.rdif)**2)
        else:
            values[0] = 1.0
    def value_shape(self):
        return ()

class detJJinv1(fenics.UserExpression):
    def __init__(self, **Kwargs) -> None:
        super().__init__(**Kwargs) 
        self.rm = 0.45
        self.r0 = 0.2
        self.rdif = self.rm - self.r0
    def eval(self, values, x):
        xn = (x[0]**2 + x[1]**2)**0.5
        if xn < self.rm and xn > self.r0-1e-6:
            values[0] = (x[0]*x[0]*self.rm/(xn**3)-1)/self.rdif
            values[1] = (x[0]*x[1]*self.rm/(xn**3))/self.rdif
            values[2] = (x[1]*x[0]*self.rm/(xn**3))/self.rdif
            values[3] = (x[1]*x[1]*self.rm/(xn**3)-1)/self.rdif
        else:
            values[0] = 0.0
            values[1] = 0.0
            values[2] = 0.0
            values[3] = 0.0
    def value_shape(self):
        return (2,2)

class detJJinv2(fenics.UserExpression):
    def __init__(self, **Kwargs) -> None:
        super().__init__(**Kwargs) 
        self.rm = 0.45
        self.r0 = 0.2
        self.rdif = self.rm - self.r0
    def eval(self, values, x):
        xn = (x[0]**2 + x[1]**2)**0.5
        if xn < self.rm and xn > self.r0-1e-6:
            values[0] = - self.rm*(x[0]*x[0]*self.r0/(xn**3)-1)/self.rdif
            values[1] = - self.rm*(x[0]*x[1]*self.r0/(xn**3))/self.rdif
            values[2] = - self.rm*(x[1]*x[0]*self.r0/(xn**3))/self.rdif
            values[3] = - self.rm*(x[1]*x[1]*self.r0/(xn**3)-1)/self.rdif
        else:
            values[0] = 1.0
            values[1] = 0.0
            values[2] = 0.0
            values[3] = 1.0
    def value_shape(self):
        return (2,2)

class detJ2(fenics.UserExpression):
    def __init__(self, **Kwargs) -> None:
        super().__init__(**Kwargs) 
        self.rm = 0.45
        self.r0 = 0.2
        self.rdif = self.rm - self.r0
    def eval(self, values, x):
        xn = (x[0]**2 + x[1]**2)**0.5
        if xn < self.rm and xn > self.r0-1e-6:
            values[0] = (1-self.rm/xn)/((self.rdif)**2)
        else:
            values[0] = 0
    def value_shape(self):
        return ()

class detJ1(fenics.UserExpression):
    def __init__(self, **Kwargs) -> None:
        super().__init__(**Kwargs) 
        self.rm = 0.45
        self.r0 = 0.2
        self.rdif = self.rm - self.r0
    def eval(self, values, x):
        xn = (x[0]**2 + x[1]**2)**0.5
        if xn < self.rm and xn > self.r0-1e-6:
            values[0] = self.rm*((self.r0+self.rm)/xn-2)/((self.rdif)**2)
        else:
            values[0] = 0
    def value_shape(self):
        return ()

class detJ0(fenics.UserExpression):
    def __init__(self, **Kwargs) -> None:
        super().__init__(**Kwargs) 
        self.rm = 0.45
        self.r0 = 0.2
        self.rdif = self.rm - self.r0
    def eval(self, values, x):
        xn = (x[0]**2 + x[1]**2)**0.5
        if xn < self.rm and xn > self.r0-1e-6:
            values[0] = (self.rm)**2*(1-self.r0/xn)/((self.rdif)**2)
        else:
            values[0] = 1.0
    def value_shape(self):
        return ()

class u_cor(fenics.UserExpression):
    def __init__(self, mu, **Kwargs) -> None:
        super().__init__(**Kwargs) 
        self.FM00 = mu[0]
        self.FM01 = mu[2]
        self.FM10 = mu[2]
        self.FM11 = mu[1]
    def eval(self, values, x):
        values[0] = self.FM00*x[0] + self.FM01*x[1]
        values[1] = self.FM10*x[0] + self.FM11*x[1]
    def value_shape(self):
        return (2,)