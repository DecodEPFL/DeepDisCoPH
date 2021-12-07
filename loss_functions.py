import torch
import torch.nn as nn


def f_loss_states(t, x, clsys, test=False):
    loss_function = nn.MSELoss(reduction='none')
    xbar = clsys.sys.xbar
    steps = t.shape[0]
    if test:
        gamma = 1
    else:
        gamma = 0.95
    loss = loss_function(x, xbar.unsqueeze(0).repeat(steps, 1, 1)).sum(dim=(1,2))
    loss = (loss * (gamma**t).flip(0)).sum()
    return loss


def f_loss_reg(clsys):
    reg = 0
    if clsys.ctl.K.shape[-1] > 1:
        for i in range(clsys.ctl.K.shape[-1]-1):
            for j in range(len(clsys.ctl.K)):
                reg = reg + \
                  torch.norm(clsys.ctl.K[j,:,:,i+1] - clsys.ctl.K[j,:,:,i]) + \
                  torch.norm(clsys.ctl.b[j,0,:,i+1] - clsys.ctl.b[j,0,:,i])
    return reg


def f_loss_u(t, xi, clsys):
    if clsys.ctl._get_name() == 'Controller':
        output_c = torch.zeros(len(t)-1, clsys.ctl.n//2, 1)
        for i in range(len(t)-1):
            output_c[i, :, :] = clsys.ctl.output(t[i], xi[i,:,:])
        loss_u = (output_c**2).sum()
    else:  # clsys.ctl._get_name() == 'MLP_Controller'
        import torch.nn.functional as F
        output_c = torch.zeros(len(t)-1, clsys.ctl.n_in, 1)
        for i in range(len(t) - 1):
            y_s = F.linear(clsys.sys.gradH(t[i], xi[i,:]).T, clsys.sys.g(t[i], xi[i,:]).T)
            output_c[i, :, :] = -clsys.ctl.forward(t[i], y_s).transpose(0,1)
        loss_u = (output_c ** 2).sum()
    return loss_u


def f_loss_ca(x, clsys, min_dist=0.5):
    steps = x.shape[0]
    min_sec_dist = 1.4 * min_dist
    # collision avoidance:
    deltax = x[:, 2::4].repeat(1, 1, clsys.sys.n // 4) - x[:, 2::4].transpose(1, 2).repeat(1, clsys.sys.n // 4, 1)
    deltay = x[:, 3::4].repeat(1, 1, clsys.sys.n // 4) - x[:, 3::4].transpose(1, 2).repeat(1, clsys.sys.n // 4, 1)
    distance_sq = deltax ** 2 + deltay ** 2
    mask = torch.logical_not(torch.eye(clsys.sys.n // 4)).unsqueeze(0).repeat(steps, 1, 1)
    loss_ca = (1 / (distance_sq + 1e-3) * (distance_sq.detach() < (min_sec_dist ** 2)) * mask).sum() / 2
    return loss_ca

