import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


def plot_trajectories(x, xbar, n_agents, text="", save=False, filename=None):
    T = 100
    plt.figure()
    plt.title(text + r': $xy$-plane - Evolution: $\star\, \rightarrow \, \circ$')
    plt.xlabel(r'$q_x$')
    plt.ylabel(r'$q_y$')
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', '#90ee90', '#c20078']
    for i in range(n_agents):
        plt.plot(x[:T,4*i+2,0].detach(), x[:T,4*i+3,0].detach(), color=colors[i%12], linewidth=1)
        plt.plot(x[T:,4*i+2,0].detach(), x[T:,4*i+3,0].detach(), color=colors[i%12], linestyle='dotted', linewidth=0.5)
    for i in range(n_agents):
        plt.plot(x[0,4*i+2,0].detach(), x[0,4*i+3,0].detach(), color=colors[i%12], marker='*')
        plt.plot(xbar[4*i+2,0].detach(), xbar[4*i+3,0].detach(), color=colors[i%12], marker='o', fillstyle='none')
    if save:
        plt.savefig('figures/' + filename+'_'+text+'_trajectories.eps', format='eps')
    else:
        plt.show()


def plot_traj_vs_time(t, n_agents, x, xi=None, text="", save=False, filename=None):
    T = 100
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', '#90ee90', '#c20078']
    plt.figure()
    plt.suptitle(text+r': $q$ (top) and $p$ (bottom) trajectories')
    plt.subplot(221)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q_x$')
    legend = []
    for i in range(n_agents):
        plt.plot(t[:T], x[:T,4*i+2,0].detach(), color=colors[i%12])
        legend.append("$q_{x,%i}(t)$" % (i+1))
    for i in range(n_agents):
        plt.plot(t[T:], x[T:,4*i+2,0].detach(), color=colors[i%12], linestyle='dotted')
    plt.legend(legend)
    plt.subplot(222)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$q_y$')
    legend = []
    for i in range(n_agents):
        plt.plot(t[:T], x[:T,4*i+3,0].detach(), color=colors[i%12])
        legend.append("$q_{y,%i}(t)$" % (i+1))
    for i in range(n_agents):
        plt.plot(t[T:], x[T:,4*i+3,0].detach(), color=colors[i%12], linestyle='dotted')
    plt.legend(legend)
    plt.subplot(223)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_x$')
    legend = []
    for i in range(n_agents):
        plt.plot(t[:T], x[:T,4*i, 0].detach(), color=colors[i%12])
        legend.append("$p_{x,%i}(t)$" % (i+1))
    for i in range(n_agents):
        plt.plot(t[T:], x[T:,4*i, 0].detach(), color=colors[i%12], linestyle='dotted')
    plt.legend(legend)
    plt.subplot(224)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_y$')
    legend = []
    for i in range(n_agents):
        plt.plot(t[:T], x[:T, 4*i+1, 0].detach(), color=colors[i%12])
        legend.append("$p_{y,%i}(t)$" % (i+1))
    for i in range(n_agents):
        plt.plot(t[T:], x[T:, 4*i+1, 0].detach(), color=colors[i%12], linestyle='dotted')
    plt.legend(legend)
    if save:
        plt.savefig('figures/' + filename+'_'+text+'_x.eps', format='eps')
    else:
        plt.show()
    # xi trajectories
    if xi is not None:
        plt.figure()
        plt.suptitle(text + r'$\xi^q$ (top) and $\xi^p$ (bottom) trajectories')
        plt.subplot(221)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\xi^q_1$')
        legend = []
        for i in range(n_agents):
            plt.plot(t, xi[:, 4*i+2, 0].detach(), color=colors[i%12])
            legend.append(r'$\xi^q_{1,%i}(t)$' % (i+1))
        plt.legend(legend)
        plt.subplot(222)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\xi^q_2$')
        legend = []
        for i in range(n_agents):
            plt.plot(t, xi[:, 4*i+3, 0].detach(), color=colors[i%12])
            legend.append(r'$\xi^q_{2,%i}(t)$' % (i+1))
        plt.legend(legend)
        plt.subplot(223)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\xi^p_1$')
        legend = []
        for i in range(n_agents):
            plt.plot(t, xi[:, 4*i, 0].detach(), color=colors[i%12])
            legend.append(r'$\xi^p_{1,%i}(t)$' % (i+1))
        plt.legend(legend)
        plt.subplot(224)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\xi^p_2$')
        legend = []
        for i in range(n_agents):
            plt.plot(t, xi[:, 4*i+1, 0].detach(), color=colors[i%12])
            legend.append(r'$\xi^p_{2,%i}(t)$' % (i + 1))
        plt.legend(legend)
        if save:
            plt.savefig('figures/' + filename+'_'+text+'_xi.eps', format='eps')
        else:
            plt.show()


def plot_energy(t, x, xi, sys, ctl, save=False, filename=None):
    plt.figure()
    plt.title(r'Energy evolution')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$H(t)$')
    H_s = torch.zeros(x.shape[0])
    H_c = torch.zeros(x.shape[0])
    for i in range(len(t)):
        H_s[i] = sys.H(t[i], x[i, :, :])
        H_c[i] = ctl.H(t[i], xi[i, :, :])
    plt.plot(t, H_s.detach())
    plt.plot(t, H_c.detach())
    plt.plot(t, H_s.detach()+H_c.detach())
    plt.legend(["sys", "ctl", "total"])
    if save:
        plt.savefig('figures/' + filename + '_energy.eps', format='eps')
    else:
        plt.show()


def plot_input(t, xi, ctl, save=False, filename=None):
    output_c = torch.zeros(xi.shape[0], ctl.n//2, 1)
    for i in range(len(t)):
        output_c[i, :, :] = ctl.output(t[i], xi[i,:,:])
    plt.figure()
    plt.title(r'Input signal')
    legendx = []
    legendy = []
    plt.subplot(121)
    for i in range(0,ctl.n//2,2):
        plt.plot(t, output_c.detach()[:,i,0])
        legendx.append(r'$u_{%i}(t)$' % (2*i+1))
    plt.legend(legendx)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$u_x(t)$')
    plt.subplot(122)
    for i in range(0, ctl.n // 2, 2):
        plt.plot(t, output_c.detach()[:, i+1, 0])
        legendy.append(r'$u_{%i}(t)$' % (2*i+2))
    plt.legend(legendy)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$u_y(t)$')
    if save:
        plt.savefig('figures/' + filename + '_input.eps', format='eps')
    else:
        plt.show()


def plot_rel_distance(t, x, min_dist, save=False, filename=None):
    deltax = x[:, 2::4].repeat(1, 1, x.shape[1]//4) - x[:, 2::4].transpose(1, 2).repeat(1, x.shape[1]//4, 1)
    deltay = x[:, 3::4].repeat(1, 1, x.shape[1]//4) - x[:, 3::4].transpose(1, 2).repeat(1, x.shape[1]//4, 1)
    distance_sq = (deltax**2 + deltay**2)
    plt.figure()
    for i in range(x.shape[1]//4):
        for j in range(x.shape[1]//4):
            if i < j:
                plt.plot(t, torch.sqrt(distance_sq[:, i, j].detach()))
    min_dist = torch.tensor([min_dist])
    plt.plot(t, min_dist.repeat(x.shape[0]), 'k--')
    plt.xlabel(r'$t$')
    plt.ylabel(r'distances')
    plt.title("Distances between any two agents")
    if save:
        plt.savefig('figures/' + filename + '_distances.eps', format='eps')
    else:
        plt.show()


def plot_grads(end_layer=-1, save=False, filename=None):
    bsm = torch.load('bsm.pt')
    max_iteration, n, _, steps = bsm.shape
    if 1 < end_layer < steps:
        steps = end_layer
        sub_i = str(int(steps))
    else:
        sub_i = 'N'
    bsm_norm = torch.zeros(max_iteration, steps)
    for epoch in range(max_iteration):
        for j in range(steps - 1):
            bsm_ = bsm[epoch, :, :, j]
            for i in range(j+1, steps - 1):
                bsm_ = torch.matmul(bsm_, bsm[epoch, :, :, i])
            bsm_norm[epoch, j] = torch.norm(bsm_.detach(), p=2)
    gradients_matrix_norm = bsm_norm.detach()
    x = np.linspace(0, max_iteration-1, max_iteration)
    fig, ax = plt.subplots()
    n_layers = steps-1
    # setup the normalization and the colormap
    normalize = mcolors.Normalize(vmin=1, vmax=n_layers)
    colormap = cm.get_cmap('jet', n_layers - 1)
    for ii in range(1, steps):
        j = n_layers - ii
        plt.plot(x, gradients_matrix_norm[:, j], color=colormap(normalize(j)), linewidth=0.5)
    plt.xlabel("Iterations")
    plt.ylabel(r'$\|\|\frac{\partial {\zeta}_{%s}}{\partial {\zeta}_{{%s}-\ell}}\|\|$' % (sub_i, sub_i), fontsize=12)
    # setup the colorbar
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    cb = plt.colorbar(scalarmappaple)
    cb.set_label('Depth $\ell$')
    plt.tight_layout()
    #plt.title("BSM")
    if save:
        plt.savefig('figures/' + filename + '_bsm.eps', format='eps')
        print("---! Figure saved in ./figures/")
    else:
        plt.show()
