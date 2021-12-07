import torch
import torch.nn.functional as F
import torch.nn as nn


class PHInterconnectionEnv(nn.Module):
    def __init__(self, sys, ctl, output="full"):
        super().__init__()
        """ Initialize the environment. Here we represent the interconnection
        between a PH system and a PH controller.
        """
        self.sys = sys
        self.ctl = ctl
        self.n = self.sys.n + self.ctl.n
        if output == "full" or output == "sys" or output == "ctl":
            self.output = output
        else:
            self.output = "full"

    def forward(self, t, z):
        x, xi = z[0:self.sys.n].clone(), z[self.sys.n:].clone()
        dHdz = self.gradH(t, z)
        dHdx, dHdxi = dHdz[0:self.sys.n, :].clone(), dHdz[self.sys.n:, :].clone()
        u_sys = -F.linear(dHdxi.T, self.ctl.g(t, xi).T)  # -y_c
        u_ctl = F.linear(dHdx.T, self.sys.g(t, x).T)  # y_s
        dx = F.linear(dHdx.T, self.sys.J - self.sys.R).T + F.linear(u_sys, self.sys.g(t, x)).T
        if self.ctl._get_name() == 'Controller':
            Jctl = self.ctl.J + (self.ctl.GG*self.ctl.maskGG-self.ctl.GG.transpose(0,1)*self.ctl.maskGG.transpose(0,1))
        else:
            Jctl = self.ctl.J
        dxi = F.linear(dHdxi.T, Jctl - self.ctl.r*self.ctl.R).T + F.linear(u_ctl, self.ctl.g(t, xi)).T
        return torch.cat([dx, dxi], dim=0)

    def f(self, t, z):
        return self.forward(t, z).T

    def H(self, t, z):
        x, xi = z[0:self.sys.n], z[self.sys.n:]
        return self.sys.H(t, x) + self.ctl.H(t, xi)

    def gradH(self, t, z):
        z = z.requires_grad_(True)
        return torch.autograd.grad(self.H(t, z), z, allow_unused=False, create_graph=True)[0]
        # x, xi = z[0:self.sys.n].clone(), z[self.sys.n:].clone()
        # return torch.cat([self.sys.gradH(t, x), self.ctl.gradH(t, xi)], dim=0)

    def g(self, t, z):
        x, xi = z[0:self.sys.n].clone(), z[self.sys.n:].clone()
        if self.output == "sys":
            return torch.cat([self.sys.g(t, x), torch.zeros_like(self.ctl.g(t, xi))], dim=0)
        elif self.output == "ctl":
            return torch.cat([torch.zeros_like(self.sys.g(t, x)), self.ctl.g(t, xi)], dim=0)
        else:
            return torch.block_diag(self.sys.g(t, x), self.ctl.g(t, xi))


class MLPInterconnectionEnv(nn.Module):
    def __init__(self, sys, ctl, output="full"):
        super().__init__()
        """ Initialize the environment. Here we represent the interconnection
        between a PH system and a MLP controller.
        """
        self.sys = sys
        self.ctl = ctl

    def forward(self, t, x):
        y_s = F.linear(self.sys.gradH(t, x).T, self.sys.g(t, x).T)
        u_sys = -self.ctl(t, y_s)  # -y_c
        dx = self.sys.f(t, x) + F.linear(u_sys, self.sys.g(t, x))
        return dx.T


class SystemEnv(nn.Module):
    def __init__(self, n_agents=1, xbar=None, ctls=None):
        """ Initialize the environment. Here we represent the system.
        """
        super().__init__()
        self.k = torch.tensor(1.0)
        self.b = torch.tensor(0.2)
        self.m = torch.tensor(1.0)
        J = torch.tensor([[0, 0, -1, 0],
                          [0, 0, 0, -1],
                          [1., 0, 0, 0],
                          [0, 1., 0, 0]])
        R = torch.tensor([[self.b, 0, 0, 0],
                          [0, self.b, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        self.n_agents = n_agents
        self.ni = 4
        self.n = self.ni * n_agents
        if ctls is None:
            ctls = torch.ones(1, n_agents)
            # n_of_inputs x n_masses
        self.interconnection = ctls
        self.J = torch.zeros((self.n, self.n))
        self.R = torch.zeros((self.n, self.n))
        for i in range(0, n_agents):
            self.J[self.ni*i:self.ni*(i+1), self.ni*i:self.ni*(i+1)] = J
            self.R[self.ni*i:self.ni*(i+1), self.ni*i:self.ni*(i+1)] = R
        if xbar is None:
            xbar = torch.zeros(self.n, 1)
        self.xbar = xbar

    def g(self, t, x):
        # g = torch.zeros((self.n, 2*int(self.interconnection.sum())))
        # idx = 0
        # for i, j in self.interconnection.nonzero():
        #     g[(4*j), idx] = 1
        #     g[(4*j)+1, idx+1] = 1
        #     idx += 2
        # return g
        g_agent = torch.tensor([[1.0, 0], [0, 1.0], [0, 0], [0, 0]])
        g = torch.zeros(0, 0)
        for i in range(self.n_agents):
            g = torch.block_diag(g,g_agent)
        return g

    def H(self, t, x):
        delta_x = x - self.xbar
        Q_agent = torch.diag(torch.tensor([1/self.m, 1/self.m, self.k,  self.k]))
        Q = torch.zeros((self.n, self.n))
        for i in range(self.n_agents):
            Q[self.ni*i:self.ni*(i+1), self.ni*i:self.ni*(i+1)] = Q_agent
        return 0.5 * F.linear(F.linear(delta_x.T, Q), delta_x.T)

    def gradH(self, t, x):
        x = x.requires_grad_(True)
        return torch.autograd.grad(self.H(t, x), x, allow_unused=False, create_graph=True)[0]

    def f(self, t, x):
        dHdx = self.gradH(t, x)
        return F.linear(dHdx.T, self.J - self.R)

    def forward(self, t, x):
        x = x.requires_grad_(True)
        return self.f(t, x).T


class Controller(nn.Module):
    def __init__(self, n_layers, h, r=0, n_agents=1, xbar=None, n_H_layers=1, interconnection=None, r_train=False):
        """ Initialize the environment. Here we represent the controller.
        """
        super().__init__()
        self.h = h
        self.n_agents = n_agents
        if interconnection is None:
            self.interconnection = torch.eye(n_agents)
        else:
            self.interconnection = interconnection
        self.ni = 4
        self.n = self.ni * self.n_agents
        ni2 = self.ni//2
        J_acc = torch.zeros((0, 0))
        R_acc = torch.zeros((0, 0))
        mask_acc = torch.zeros((0, 0))
        J = torch.cat((torch.cat((torch.zeros(ni2, ni2), -torch.eye(ni2)), dim=1),
                       torch.cat((torch.eye(ni2), torch.zeros(ni2, ni2)), dim=1)), dim=0)
        R = torch.cat((torch.cat((torch.eye(ni2), torch.zeros(ni2, ni2)), dim=1),
                       torch.zeros(ni2, 2*ni2)), dim=0)
        mask = torch.ones(4, 4)
        # local interconnection
        for i in range(self.n_agents):
            J_acc = torch.block_diag(J_acc, J)
            R_acc = torch.block_diag(R_acc, R)
            mask_acc = torch.block_diag(mask_acc, mask)
        # interconnection inter-controllers
        GG = torch.zeros((self.n, self.n))
        gg = torch.cat((torch.zeros(ni2, 2*ni2),
                       torch.cat((torch.eye(ni2), torch.zeros(ni2, ni2)), dim=1)), dim=0)
        maskGG = torch.zeros((self.n, self.n))
        for i, j in interconnection.nonzero():
            if not(i==j):
                maskGG[self.ni*i:self.ni*(i+1), self.ni*j:self.ni*(j+1)] = torch.ones(self.ni, self.ni)
            if i<j:
                GG[self.ni*i:self.ni*(i+1), self.ni*j:self.ni*(j+1)] = gg
            elif i>j:
                GG[self.ni*i:self.ni*(i+1), self.ni*j:self.ni*(j+1)] = -gg.transpose(0, 1)
        self.J = J_acc
        self.R = R_acc
        self.mask = mask_acc
        self.GG = nn.Parameter(GG)
        self.maskGG = maskGG
        if xbar is None:
            xbar = torch.zeros(self.n, 1)
        self.xbar = xbar
        mask_sq = self.mask.unsqueeze(2).repeat(1, 1, n_layers)
        self.K = nn.Parameter(torch.randn((n_H_layers, self.n, self.n, 1)) * mask_sq / 10)
        self.b = nn.Parameter(torch.randn((n_H_layers, 1, self.n, n_layers)) / 10)
        if r_train:
            print("r is a trainable parameter")
            self.r = nn.Parameter(torch.tensor(r))
        else:
            self.r = torch.tensor(r)
        self.mask_g_out = torch.kron(self.interconnection, torch.ones(self.ni,ni2))
        self.g_out = nn.Parameter(torch.kron(self.interconnection, torch.tensor([[1.0, 0], [0, 1.0], [0, 0], [0, 0]])))

    def g(self, t, x):
        g = self.g_out * self.mask_g_out
        # # g = torch.kron(self.interconnection, torch.tensor([[1.0, 0], [0, 1.0], [0, 0], [0, 0]]))
        # g_agent = torch.tensor([[0, 0], [0, 0], [1.0, 0], [0, 1.0]])
        # g = torch.zeros((self.n, self.ni//2 * self.n_agents))
        # for i in range(self.n_agents):
        #     g[self.ni*i:self.ni*(i+1), (self.ni//2)*i:(self.ni//2)*(i+1)] = g_agent
        return g

    def H(self, t, x):
        x = (x - self.xbar).T
        t = int(t/self.h)
        if t >= self.K.shape[3]:
            t = self.K.shape[3] - 1
        for i in range(len(self.K)):
            x = F.linear(x, self.K[i,:,:,t] * self.mask, self.b[i,:,:,t])
            h = torch.log(torch.cosh(x))
            # if torch.isinf(h).sum() > 0:
            #     h_max = torch.abs(x) - torch.log(torch.tensor(2))
            #     h = h * ~(torch.isinf(h.detach())) + h_max * torch.isinf(h.detach())
            x = h
        return x.sum()

    def gradH(self, t, x):
        x = x.requires_grad_(True)
        H = self.H(t, x)
        dHdx = torch.autograd.grad(H, x, allow_unused=False, create_graph=True)[0]
        return dHdx

    def f(self, t, x):
        dHdx = self.gradH(t, x)
        return F.linear(dHdx.T,
                        self.J+(self.GG*self.maskGG-self.GG.transpose(0,1)*self.maskGG.transpose(0,1)) - self.r*self.R)

    def forward(self, t, x):
        return self.f(t, x)

    def output(self, t, x):
        return F.linear(self.gradH(t, x).T, self.g(t, x).T).T


class MLP_Controller(nn.Module):
    def __init__(self, h, interconnection, steps=None):
        super().__init__()
        n_agents = interconnection.shape[0]
        in_size = 2
        self.n_in = n_agents * in_size
        self.h = h
        self.interconnection = interconnection
        l0, l1, l2, l3 = in_size, 4*in_size, 4*in_size, in_size
        self.mask1 = torch.kron(self.interconnection.transpose(0,1).contiguous(), torch.ones(l1, l0))
        self.mask2 = torch.kron(torch.eye(n_agents), torch.ones(l2, l1))
        self.mask3 = torch.kron(torch.eye(n_agents), torch.ones(l3, l2))
        self.w1 = nn.Parameter(self.mask1 * torch.randn(n_agents*l1, n_agents*l0)/10)
        self.b1 = nn.Parameter(torch.randn(n_agents*l1))
        self.w2 = nn.Parameter(self.mask2 * torch.randn(n_agents*l2, n_agents*l1)/10)
        self.b2 = nn.Parameter(torch.randn(n_agents*l2))
        self.w3 = nn.Parameter(self.mask3 * torch.randn(n_agents*l3, n_agents*l2)/10)
        self.b3 = nn.Parameter(torch.randn(n_agents*l3))

    def forward(self, t, x):
        x = torch.tanh(F.linear(x, self.mask1 * self.w1, self.b1))
        x = torch.tanh(F.linear(x, self.mask2 * self.w2, self.b2))
        x = F.linear(x, self.mask3 * self.w3, self.b3)
        return x
