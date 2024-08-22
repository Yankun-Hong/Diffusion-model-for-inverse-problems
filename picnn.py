import numpy as np
import torch, scipy
import torch.nn as nn
import os, pickle
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.notebook import trange

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Gradient(nn.Module):
    '''A network block that output the gradient by the displace u
        The input is (batch, 2, N*N), the channels are for displacement [ux, uy] 
        The output is (batch, 4, Ngrad):[ux_1, uy_1, ux_2, uy_2]
    '''
    def __init__(self, Gradx, Grady):
        super().__init__()
        self.Gradx = Gradx
        self.Grady = Grady
    def forward(self, x):
        gx = torch.einsum('lk, ijk -> ijl', self.Gradx, x)
        gy = torch.einsum('lk, ijk -> ijl', self.Grady, x)
        return torch.cat((gx, gy), dim=1)

class InnerEnergy(nn.Module):
    '''A network block that output the inner energy
        The input is (batch, 4, Ngrad) and (batch, N*N), the first argument is for displacement gradient, 
        the last one is for  Young's modulus 
        The output is (batch,)
    '''
    def __init__(self, CG2DG, Vitg):
        super().__init__()
        self.CG2DG = CG2DG
        self.Vitg = Vitg
    def forward(self, x, E):
        Em = torch.einsum('ij, kj -> ki', self.CG2DG, E)
        nu = 0.3
        nu1, nu2 = 1.0/(2*(1 + nu)), nu/((1 + nu)*(1 - 2*nu))
        mu, lmbda = Em*nu1, Em*nu2
        J = (1+x[:,0,:])*(1+x[:,3,:]) - x[:,1,:]*x[:,2,:]
        Ic = (1+x[:,0,:])**2 + (1+x[:,3,:])**2 + x[:,1,:]**2 + x[:,2,:]**2
        psi = 0.5*mu*(Ic - 2) - mu*torch.log(J) + 0.5*lmbda*((torch.log(J))**2)
        Psi = torch.einsum('j, ij -> i', self.Vitg, psi)
        return Psi
    
class OuterEnergy(nn.Module):
    '''A network block that output the inner energy
        The input is (batch, 2, N*N), the channels are for displacement [ux, uy] 
        The output is (batch,)
    '''
    def __init__(self, S_X, S_Y):
        super().__init__()
        self.S_X = S_X
        self.S_Y = S_Y
    def forward(self, x):
        xx = x[:,0,...].reshape((x.shape[0],-1))
        xy = x[:,1,...].reshape((x.shape[0],-1))
        ox = torch.einsum('j, ij -> i', self.S_X, xx)
        oy = torch.einsum('j, ij -> i', self.S_Y, xy)
        return ox + oy

class SED(nn.Module):
    '''A network block that output the strain energy density by the displace u and Young's modulus
        The input is (batch, 2, N, N) and (batch, N, N), the first argument is for displacement, 
        the last one is for  Young's modulus
    '''
    def __init__(self, Gradx, Grady, CG2DG, Vitg, S_X, S_Y):
        super().__init__()
        self.CG2DG = CG2DG
        self.Vitg = Vitg
        self.S_X = S_X
        self.S_Y = S_Y
        self.Grad = Gradient(Gradx, Grady)   
        self.Inner = InnerEnergy(CG2DG, Vitg)
        self.Outer = OuterEnergy(S_X, S_Y)
    def forward(self, x, E):
        xr = x.reshape((x.shape[0], x.shape[1], -1))
        Em = (95.0*E + 5.0).reshape((E.shape[0], -1))
        xg = self.Grad(xr)
        IPsi = self.Inner(xg, Em)
        OPsi = self.Outer(xr)
        return IPsi + OPsi
        
class Dirichlet(nn.Module):
    '''A network block that output the boundary sum of the Dirichlet boundary
        The input is (batch, 2, N, N)
        The output is (batch,)
    '''      
    def __init__(self, Mfix):
        super().__init__()
        self.Mfix = Mfix
    def forward(self, x):
        xr = x.reshape((x.shape[0], x.shape[1], -1))
        diri = torch.einsum('ij, klj -> kli', self.Mfix, xr)
        return torch.sum(torch.square(diri), (1,2))
    
class Observe(nn.Module):
    '''A network block that output the observation
        The input is (batch, 2, N, N)
        The output is (batch, 2, Nobs)
    '''      
    def __init__(self, Mobs):
        super().__init__()
        self.Mobs = Mobs
    def forward(self, x):
        xr = x.reshape((x.shape[0], x.shape[1], -1))
        obs = torch.einsum('ij, klj -> kli', self.Mobs, xr)
        return obs


with open(os.path.join(os.path.abspath('.'), 'KnownData', 'Losscomp' + '.pickle'), 'rb') as file:
    A2N, N2A, Ma2n, Mn2a, Mobs, Mfix, CG2DG, Gradx, Grady, Vitg, S_X, S_Y = pickle.load(file)

Mobst = torch.from_numpy(Mobs.todense()).float().to(device)
Mfixt = torch.from_numpy(Mfix.todense()).float().to(device)
CG2DGt = torch.from_numpy(CG2DG).float().to(device)
Gradxt = torch.from_numpy(Gradx).float().to(device)
Gradyt = torch.from_numpy(Grady).float().to(device)
Vitgt = torch.from_numpy(Vitg).float().to(device)
S_Xt = torch.from_numpy(S_X).float().to(device)
S_Yt = torch.from_numpy(S_Y).float().to(device)

sed = SED(Gradxt, Gradyt, CG2DGt, Vitgt, S_Xt, S_Yt).to(device)
diri = Dirichlet(Mfixt).to(device)
obs = Observe(Mobst).to(device)

def loss_sed(u, E):
    Es = torch.squeeze(E, 1)
    ls = sed(u*0.01, Es)*100
    ld = diri(u)
    return torch.mean(ls) + torch.mean(ld)

def loss_sup(u, u_ref, u_obs):
    la = (u-u_ref)**2
    la = torch.mean(la.reshape(la.shape[0], -1), dim=-1)
    lobs = (obs(u)-u_obs)**2
    lobs = torch.mean(lobs.reshape(lobs.shape[0], -1), dim=-1)
    return torch.mean(la+lobs) 

class BiasLayer(torch.nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        bias_value = torch.randn(shape)
        self.bias_layer = torch.nn.Parameter(bias_value)
    
    def forward(self, x):
        return x + self.bias_layer[None, ...]

class PICNN(nn.Module):
    '''A convolution neural network to approximate the forward model.'''
    def __init__(self, resolution=32, channels=[1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 8, 8, 4, 4, 2, 2], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.N = resolution
        self.Np = resolution*resolution
        self.channels = channels
        #self.fc1 = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1,
        #                     padding=1, bias=True)
        self.fc1 = BiasLayer((channels[0], self.N, self.N))
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.res1 = nn.Conv2d(channels[0], channels[2], kernel_size=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels[2], channels[3], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.conv4 = nn.Conv2d(channels[3], channels[4], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.res2 = nn.Conv2d(channels[2], channels[4], kernel_size=1, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(channels[4], channels[5], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.conv6 = nn.Conv2d(channels[5], channels[6], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.res3 = nn.Conv2d(channels[4], channels[6], kernel_size=1, stride=1)
        self.pool6 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(channels[6], channels[7], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.conv8 = nn.Conv2d(channels[7], channels[8], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.res4 = nn.Conv2d(channels[6], channels[8], kernel_size=1, stride=1)
        self.pool8 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(channels[8], channels[9], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.conv10 = nn.Conv2d(channels[9], channels[10], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.res5 = nn.Conv2d(channels[8], channels[10], kernel_size=1, stride=1)
        self.pool10 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(channels[10], channels[11], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.conv12 = nn.Conv2d(channels[11], channels[12], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.res6 = nn.Conv2d(channels[10], channels[12], kernel_size=1, stride=1)
        self.pool12 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(channels[12], channels[13], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.conv14 = nn.Conv2d(channels[13], channels[14], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.res7 = nn.Conv2d(channels[12], channels[14], kernel_size=1, stride=1)
        self.pool14 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv2d(channels[14], channels[15], kernel_size=9, stride=1, 
                               padding=4, bias=True)
        self.fc16 = nn.Conv2d(channels[15], channels[15], kernel_size=1, stride=1,
                              padding=0, bias=True)
        #self.fc16 = BiasLayer((channels[15], self.N, self.N))
        self.act = nn.SiLU()
    def forward(self, x):
        y = self.act(self.fc1(x))
        y_ = self.conv2(self.act(self.conv1(y)))
        y = self.act(y_ + self.res1(y))
        # y = self.pool2(y)
        y_ = self.conv4(self.act(self.conv3(y)))
        y = self.act(y_ + self.res2(y))
        # y = self.pool4(y)
        y_ = self.conv6(self.act(self.conv5(y)))
        y = self.act(y_ + self.res3(y))
        # y = self.pool6(y)
        y_ = self.conv8(self.act(self.conv7(y)))
        y = self.act(y_ + self.res4(y))
        # y = self.pool8(y)
        y_ = self.conv10(self.act(self.conv9(y)))
        y = self.act(y_ + self.res5(y))
        # y = self.pool10(y)
        y_ = self.conv12(self.act(self.conv11(y)))
        y = self.act(y_ + self.res6(y))
        # y = self.pool12(y)
        y_ = self.conv14(self.act(self.conv13(y)))
        y = self.act(y_ + self.res7(y))
        # y = self.pool14(y)
        y = self.act(self.conv15(y))
        y= self.fc16(y)
        return y

class FM():
    '''The forward model'''
    def __init__(self, pdeloss, suploss, Mobst, config) -> None:
        self.network = PICNN().to(device)
        self.obs = Observe(Mobst).to(device)
        self.pdeloss = pdeloss
        self.suploss = suploss
        self.n_epochsup = config['n_epochsup']
        self.n_epochuns = config['n_epochuns']
        self.n_epochsem = config['n_epochsem']
        self.batch_size = config['batch_size']
        self.lr = config['learning_rate']
        self.lr_s = config['learning_rate_sem']
        self.config = config
    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)
    def load_para_sup(self):
        self.network.load_state_dict(torch.load(os.path.join(os.path.abspath('.'), 'NNfw_para_sup.pth')))
        self.network.eval()
    def load_para_uns(self):
        self.network.load_state_dict(torch.load(os.path.join(os.path.abspath('.'), 'NNfw_para_uns.pth')))
        self.network.eval()
    def load_para(self):
        self.network.load_state_dict(torch.load(os.path.join(os.path.abspath('.'), 'NNfw_para.pth')))
        self.network.eval()
    def evaluate(self, E):
        self.network.eval()
        if E.dim() == 2:
            Et  = E[None,None,...]
        elif E.dim() == 3:
            Et = E[:,None,...]
        else:
            Et = E
        return self.network(Et)*0.01
    def observe(self, E):
        u = self.evaluate(E)
        return self.obs(u)
    def derivative(self, E):
        self.network.eval()
        Et = E.reshape(-1)
        Et.requires_grad_(True)
        Er = Et.reshape((1,1,self.network.N, self.network.N))
        J = torch.empty(self.obs.Mobs.shape[0], self.network.Np)
        obs = self.obs(self.network(Er))
        obs.reshape(-1)
        for i in range(obs.shape[0]):
            obs[0,i].backward(retain_graph=True)
            t = Et.grad
            J[i,:] = t[0,:]
            Et.grad.data.zero_()
        return J
    def supervised_train(self, dataset):
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        optimizer = Adam(self.network.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.config['lr_decay'], patience=120, 
                                      threshold=self.config['lr_threshold'], threshold_mode='rel', 
                                      cooldown=200, min_lr=self.config['lr_min'])
        tqdm_epoch = trange(self.n_epochsup)
        
        self.network.train()
        for epoch in tqdm_epoch:
            avg_loss = 0.
            num_items = 0
            for x, y, z in data_loader:
                x = torch.tensor(x, device=device) # x.to(device) 
                y = torch.tensor(y, device=device)
                z = torch.tensor(z, device=device)
                yp = self.network(x)   
                loss = self.suploss(yp, y, z)
                optimizer.zero_grad()
                loss.backward()    
                optimizer.step()
                scheduler.step(loss)
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
            # Print the averaged training loss so far.
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items * 1000))
            # Update the checkpoint after each epoch of training.
            torch.save(self.network.state_dict(), 'NNfw_para_sup.pth')
        self.network.eval() 
    def unsupervised_train(self, dataset):
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        optimizer = Adam(self.network.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.config['lr_decay'], patience=120, 
                                      threshold=self.config['lr_threshold'], threshold_mode='rel', 
                                      cooldown=200, min_lr=self.config['lr_min'])
        tqdm_epoch = trange(self.n_epochuns)
        
        self.network.train()
        n = 0
        for epoch in tqdm_epoch:
            avg_loss = 0.
            num_items = 0
            for x, y in data_loader:
                n+=1
                x = torch.tensor(x, device=device) # x.to(device)   
                yp = self.network(x)    
                loss = self.pdeloss(yp, x)
                optimizer.zero_grad()
                loss.backward()    
                optimizer.step()
                scheduler.step(loss)
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
            # Print the averaged training loss so far.
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items * 10))
            # Update the checkpoint after each epoch of training.
            torch.save(self.network.state_dict(), 'NNfw_para_uns.pth')
        self.network.eval()   
    def semisupervised_train(self, dataset_sup, dataset_uns, if_pretrained = True):
        data_loader_sup = DataLoader(dataset_sup, batch_size=self.batch_size, shuffle=True, num_workers=4)
        data_loader_uns = DataLoader(dataset_uns, batch_size=self.batch_size, shuffle=True, num_workers=4)

        if if_pretrained:
            optimizer = Adam(self.network.parameters(), lr=self.lr_s)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.config['lr_decay'], patience=120, 
                                        threshold=self.config['lr_threshold'], threshold_mode='rel', 
                                        cooldown=200, min_lr=self.config['lr_min_sem'])
        else:
            optimizer = Adam(self.network.parameters(), lr=self.lr)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.config['lr_decay'], patience=120, 
                                        threshold=self.config['lr_threshold'], threshold_mode='rel', 
                                        cooldown=200, min_lr=self.config['lr_min'])
        tqdm_epoch = trange(self.n_epochsem)
        
        self.network.train()
        dataloader_iterator = iter(data_loader_sup)
        for epoch in tqdm_epoch:
            avg_loss = 0.
            avg_loss1 = 0.
            num_items = 0
            for x, y in data_loader_uns:
                try:
                    x1, y1, z1 = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(data_loader_sup)
                    x1, y1, z1 = next(dataloader_iterator)
                x = torch.tensor(x, device=device) # x.to(device) 
                x1 = torch.tensor(x1, device=device)   
                y1 = torch.tensor(y1, device=device)   
                z1 = torch.tensor(z1, device=device)  
                yp = self.network(x)  
                yp1 = self.network(x1)  
                loss = self.pdeloss(yp, x)
                loss1 = self.suploss(yp1, y1, z1)
                losst = 0.1*loss + 1*loss1
                optimizer.zero_grad()
                losst.backward()    
                optimizer.step()
                scheduler.step(losst)
                avg_loss += loss.item() * x.shape[0]
                avg_loss1 += loss1.item() * x1.shape[0]
                num_items += x.shape[0]
            tqdm_epoch.set_description('Average Loss: {:5f} = {:5f} + {:5f}'.format(
                (avg_loss+avg_loss1)/num_items*1000, avg_loss/num_items*10, avg_loss1/num_items*1000))
            torch.save(self.network.state_dict(), 'NNfw_para.pth')
        self.network.eval() 

configfw = {  
            'n_epochsup': 1500, # number of training epochs for supervised learning
            'n_epochuns': 250, # number of training epochs for unsupervised learning
            'n_epochsem': 400, # number of training epochs for semisupervised learning
            'batch_size': 32, # size of a mini-batch
            'learning_rate': 1.8e-3, # learning rate
            'learning_rate_sem': 5e-5, # learning rate for semisupervised learning
            'ema_decay': 0.999, # decay rate for Exponential Moving Average 
            'lr_decay': 0.9,
            'lr_threshold': 1e-5,
            'lr_min': 5e-5,
            'lr_min_sem': 5e-6
            }

Forward_Model = FM(loss_sed, loss_sup, Mobst, configfw)               