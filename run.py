#!/usr/bin/env python
"""
Train a controller for the system of 12 robots
Author: Clara Galimberti (clara.galimberti@epfl.ch)
Usage:
python run.py                --model         [MODEL]             \
                             --with_ca       [CA_TASK]           \
Flags:
  --model: model to use for the controller. Available options: distributed_HDNN, distributed_HDNN_TI, distributed_MLP.
  --with_ca: whether to train the controllers with the collision avoidance task.
"""

import torch
from torchdiffeq import odeint
import argparse

from train import train_HDNN_TI, train_HDNN_TV, train_MLP
from plots import plot_trajectories, plot_traj_vs_time, plot_rel_distance
from loss_functions import f_loss_states, f_loss_u, f_loss_ca


def main(model, with_ca):
    """
    :param model: Model to be trained. Select from: "distributed_HDNN", "distributed_HDNN_TI", "distributed_MLP".
    :param with_ca: Train model with collision avoidance task.
    :return:
    """
    # # # # # # # # Parameters # # # # # # # #
    n_agents = 12  # agents are not interconnected (even when having a controller). Each of them acts independently
    t_end = 5
    steps = 101
    min_dist = 0.5  # min distance for collision avoidance
    # # # # # # # # Interconnection # # # # # # # #
    s = torch.eye(n_agents) + torch.diag(torch.ones(n_agents - 1), 1) + torch.diag(torch.ones(n_agents - 1), -1)
    s[0, -1] = 1
    s[-1, 0] = 1
    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate = 1e-2
    alpha_u = 0.5  # Regularization parameter for penalizing the input
    if with_ca:
        alpha_ca = 100.0  # Collision avoidance regularization factor
    else:
        alpha_ca = 0
    # # # # # # # # Training # # # # # # # #
    if model == "distributed_HDNN":
        out = train_HDNN_TV(n_agents, t_end, steps, min_dist, s, learning_rate, alpha_u, alpha_ca)
    elif model == "distributed_HDNN_TI":
        out = train_HDNN_TI(n_agents, t_end, steps, min_dist, s, learning_rate, alpha_u, alpha_ca)
    elif model == "distributed_MLP":
        out = train_MLP(n_agents, t_end, steps, min_dist, s, learning_rate, alpha_u, alpha_ca)
    else:
        raise ValueError("Model not implemented")
    clsys, z0, t = out
    if model == "distributed_MLP":
        x0 = z0
    else:
        x0, xi0 = z0[0:clsys.sys.n], z0[clsys.sys.n:]
    # # # # # # # # Print & plot results # # # # # # # #
    print("------------------------------------")
    save = True
    filename = model
    z = odeint(clsys, z0, t)
    if model == "distributed_MLP":
        x = z
        xi = z
    else:
        x, xi = z[:, 0:clsys.sys.n], z[:, clsys.sys.n:]
    plot_trajectories(x, clsys.sys.xbar, clsys.sys.n_agents, text="CL", save=save, filename=filename)
    plot_traj_vs_time(t, clsys.sys.n_agents, x, text="CL", save=save, filename=filename)
    x_ol = odeint(clsys.sys, x0, t)
    plot_trajectories(x_ol, clsys.sys.xbar, clsys.sys.n_agents, text="OL", save=save, filename=filename)
    plot_traj_vs_time(t, clsys.sys.n_agents, x_ol, text="OL", save=save, filename=filename)
    # Relative distance:
    if not (alpha_ca == 0):
        plot_rel_distance(t, x, min_dist, save=save, filename=filename)
        # Number of collisions
        deltax = x[:, 2::4].repeat(1, 1, x.shape[1] // 4) - x[:, 2::4].transpose(1, 2).repeat(1, x.shape[1] // 4, 1)
        deltay = x[:, 3::4].repeat(1, 1, x.shape[1] // 4) - x[:, 3::4].transpose(1, 2).repeat(1, x.shape[1] // 4, 1)
        distance_sq = (deltax ** 2 + deltay ** 2)
        n_coll = ((0.0001 < distance_sq) * (distance_sq < 0.25)).sum()
        print("Number of collisions after training: %d" % n_coll)
        # Collisions after perturbation on initial conditions:
        n_perturbations = 100
        n_coll = 0
        for i in range(n_perturbations):
            with torch.no_grad():
                dx_pert = (0.5 * torch.rand(clsys.sys.n // 4, 1).repeat(1, 4) *
                           torch.nn.functional.normalize(torch.rand(clsys.sys.n//4,4))).reshape(clsys.sys.n,1)
                if model == "distributed_MLP":
                    dz_pert = dx_pert
                else:
                    dz_pert = torch.zeros_like(z0)
                    dz_pert[0:clsys.sys.n] = dx_pert
            z_pert = odeint(clsys, z0+dz_pert, t, method='euler', options={'step_size': t_end/(steps-1)})
            x_pert, xi_pert = z_pert[:, 0:clsys.sys.n], z_pert[:, clsys.sys.n:]
            deltax = x_pert[:,2::4].repeat(1,1, x.shape[1]//4) - x_pert[:,2::4].transpose(1,2).repeat(1,x.shape[1]//4,1)
            deltay = x_pert[:,3::4].repeat(1,1, x.shape[1]//4) - x_pert[:,3::4].transpose(1,2).repeat(1,x.shape[1]//4,1)
            distance_sq = (deltax ** 2 + deltay ** 2)
            n_coll += ((0.0001 < distance_sq) * (distance_sq < 0.25)).sum()
        n_coll = n_coll / n_perturbations
        print("Averaged number of collisions when perturbed: %.2f" % n_coll)
        print("------------------------------------")
    # T_s = T
    loss_nom = f_loss_states(t, x, clsys, test=True)
    loss_u_nom = alpha_u * f_loss_u(t, xi, clsys)
    loss_ca_nom = alpha_ca * f_loss_ca(x, clsys, min_dist=min_dist)
    loss_total_nom = loss_nom + loss_u_nom + loss_ca_nom
    # T_s = 10 T
    t_end = 10 * t_end
    steps = 10 * (steps - 1) + 1
    t = torch.linspace(0, t_end, steps)
    z = odeint(clsys, z0, t)
    if model == "distributed_MLP":
        x = z
        xi = z
    else:
        x, xi = z[:, 0:clsys.sys.n], z[:, clsys.sys.n:]
    plot_trajectories(x, clsys.sys.xbar, clsys.sys.n_agents, text="CL_extended", save=save, filename=filename)
    plot_traj_vs_time(t, clsys.sys.n_agents, x, text="CL_extended", save=save, filename=filename)
    loss_ext = f_loss_states(t, x, clsys, test=True)
    loss_u_ext = alpha_u * f_loss_u(t, xi, clsys)
    loss_ca_ext = alpha_ca * f_loss_ca(x, clsys, min_dist=min_dist)
    loss_total_ext = loss_ext + loss_u_ext + loss_ca_ext
    print("\t\t|\t T = T_s \t|\t T = 10 T_s \n" +
          "Loss \t\t|\t %7.1f \t|\t %7.1f \n" % (loss_nom, loss_ext) +
          "Loss_u \t\t|\t %7.1f \t|\t %7.1f \n" % (loss_u_nom, loss_u_ext) +
          "Loss_ca \t|\t %7.1f \t|\t %7.1f \n" % (loss_ca_nom, loss_ca_ext) +
          "Total \t\t|\t %7.1f \t|\t %7.1f \n" % (loss_total_nom, loss_total_ext)
          )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='distributed_HDNN')
    parser.add_argument('--with_ca', type=bool, default=True)
    args = parser.parse_args()
    # Run main
    main(args.model, args.with_ca)

