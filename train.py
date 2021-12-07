import torch
from torchdiffeq import odeint

from envCT import PHInterconnectionEnv, SystemEnv, Controller
from envCT import MLPInterconnectionEnv, MLP_Controller
from params import set_initial_conditions
from loss_functions import f_loss_states, f_loss_reg, f_loss_u, f_loss_ca


def train_HDNN(n_agents, t_end, steps, min_dist, s, learning_rate, alpha_u, alpha_ca, n_H_layers, n_layers,
               grad_info=False):
    # # # # # # # # Hyperparameters # # # # # # # #
    n_H_layers = n_H_layers
    max_iteration = 300
    alpha = 125.0-alpha_ca  # Regularization parameter (between consecutive layers)
    r = 12.0  # indicates how much dissipation the controller has
    # # # # # # # # Step size # # # # # # # #
    h = t_end/(steps-1)
    # # # # # # # # System # # # # # # # #
    # Initial and final system state values
    x0, xbar = set_initial_conditions(n_agents)
    sys = SystemEnv(n_agents, xbar, ctls=s)  # pre-stabilized system
    # # # # # # # # Controller # # # # # # # #
    ctl = Controller(n_layers=n_layers, h=h, r=r, n_agents=n_agents, interconnection=s, n_H_layers=n_H_layers)
    # Initial controller state values
    xi0 = torch.zeros(ctl.n, 1)
    xi0[::4] = torch.ones(ctl.n//4, 1) * 3
    # # # # # # # # Closed loop system # # # # # # # #
    clsys = PHInterconnectionEnv(sys, ctl)
    # # # # # # # # Define optimizer, loss and parameters # # # # # # # #
    t = torch.linspace(0, t_end, steps)
    optimizer = torch.optim.Adam(clsys.parameters(), lr=learning_rate)
    # # # # # # # # Define BSM # # # # # # # #
    if grad_info:
        bsm = torch.zeros(max_iteration, clsys.n, clsys.n, t.shape[0])
    # # # # # # # # Start training # # # # # # # #
    z0 = torch.cat([x0, xi0], dim=0)
    for epoch in range(max_iteration):
        optimizer.zero_grad()
        if grad_info:
            z0.requires_grad_(True)
        z = odeint(clsys, z0, t, method='euler', options={'step_size': h})
        if grad_info:
            for j in range(t.shape[0]-1):
                z0_ = z[j,:].clone().detach()
                z0_.requires_grad_(True)
                t_ = torch.tensor([t[j], t[j+1]])
                z_ = odeint(clsys, z0_, t_, method='euler', options={'step_size': h})
                bsm_ = torch.zeros(clsys.n, clsys.n)
                for i in range(clsys.n):
                    v = torch.zeros(1, clsys.n)
                    v[0, i] = 1
                    bsm_[:, i:i+1] = torch.autograd.grad(torch.matmul(v,z_[-1,:,:]), z0_, allow_unused=False,
                                                         create_graph=True)[0]
                bsm[epoch,:,:,j] = bsm_
                del bsm_
        x, xi = z[:, 0:clsys.sys.n], z[:, clsys.sys.n:]
        if z[-1, :].isnan().sum():
            print("Break in iteration %i" % epoch)
            continue
        loss = f_loss_states(t, x, clsys)
        reg = f_loss_reg(clsys)
        loss_u = f_loss_u(t, xi, clsys)
        loss_ca = f_loss_ca(x, clsys, min_dist=min_dist)
        # print all losses
        if not grad_info:
            print("Iteration: %i \t|\t " % epoch +
                  "Loss: %.1f \t|\t " % loss +
                  "Reg: %.1f \t|\t " % (alpha*reg) +
                  "loss_u: %.1f \t|\t " % (alpha_u*loss_u) +
                  "loss_ca: %.1f \t|\t " % (alpha_ca*loss_ca)
                  )
        (loss + alpha*reg + alpha_u*loss_u + alpha_ca*loss_ca).backward()
        optimizer.step()
    # # # # # # # # Gradients # # # # # # # #
    if grad_info:
        torch.save(bsm, 'bsm.pt')
    return clsys, z0, t


def train_HDNN_TI(n_agents, t_end, steps, min_dist, s, learning_rate, alpha_u, alpha_ca):
    torch.manual_seed(100)
    n_H_layers = 5
    n_layers = 1
    out = train_HDNN(n_agents, t_end, steps, min_dist, s, learning_rate, alpha_u, alpha_ca, n_H_layers, n_layers)
    return out


def train_HDNN_TV(n_agents, t_end, steps, min_dist, s, learning_rate, alpha_u, alpha_ca, grad_info=False):
    torch.manual_seed(1000)
    n_H_layers = 1
    n_layers = steps
    out = train_HDNN(n_agents, t_end, steps, min_dist, s, learning_rate, alpha_u, alpha_ca, n_H_layers, n_layers,
                     grad_info=grad_info)
    return out


def train_MLP(n_agents, t_end, steps, min_dist, s, learning_rate, alpha_u, alpha_ca):
    torch.manual_seed(3)
    # # # # # # # # Hyperparameters # # # # # # # #
    max_iteration = 600
    # # # # # # # # Step size # # # # # # # #
    h = t_end/(steps-1)
    # # # # # # # # System # # # # # # # #
    # Initial and final system state values
    x0, xbar = set_initial_conditions(n_agents)
    sys = SystemEnv(n_agents, xbar, ctls=s)  # system with guidance law
    # # # # # # # # Controller # # # # # # # #
    ctl = MLP_Controller(h, s)
    clsys = MLPInterconnectionEnv(sys, ctl)
    # # # # # # # # Define optimizer, loss and parameters # # # # # # # #
    t = torch.linspace(0, t_end, steps)
    optimizer = torch.optim.Adam(clsys.parameters(), lr=learning_rate)
    # # # # # # # # Start training # # # # # # # #
    for epoch in range(max_iteration):
        optimizer.zero_grad()
        x = odeint(clsys, x0, t, method='euler', options={'step_size': h})
        loss = f_loss_states(t, x, clsys)
        loss_u = f_loss_u(t, x, clsys)
        loss_ca = f_loss_ca(x, clsys, min_dist=min_dist)
        # print all losses
        print("Iteration: %i \t|\t " % epoch +
              "Loss: %.1f \t|\t " % loss +
              "loss_u: %.1f \t|\t " % (alpha_u * loss_u) +
              "loss_ca: %.1f \t|\t " % (alpha_ca * loss_ca)
              )
        (loss + alpha_u*loss_u + alpha_ca*loss_ca).backward()
        optimizer.step()
    return clsys, x0, t
