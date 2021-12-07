import torch


def set_initial_conditions(n_agents):
    if n_agents == 12:
        x0 = torch.tensor([[0], [0.5], [-3], [5],
                           [0], [0.5], [-3], [3],
                           [0], [0.5], [-3], [1],
                           [0], [-0.5], [-3], [-1],
                           [0], [-0.5], [-3], [-3],
                           [0], [-0.5], [-3], [-5],
                           # second column
                           [-0], [0.5], [3], [5],
                           [-0], [0.5], [3], [3],
                           [-0], [0.5], [3], [1],
                           [0], [-0.5], [3], [-1],
                           [0], [-0.5], [3], [-3],
                           [0], [-0.5], [3], [-5],
                           ])
        xbar = torch.tensor([[0], [0], [3], [-5],
                             [0], [0], [3], [-3],
                             [0], [0], [3], [-1],
                             [0], [0], [3], [1],
                             [0], [0], [3], [3],
                             [0], [0], [3], [5],
                             # second column
                             [0], [0], [-3], [-5],
                             [0], [0], [-3], [-3],
                             [0], [0], [-3], [-1],
                             [0], [0], [-3], [1],
                             [0], [0], [-3], [3],
                             [0], [0], [-3], [5.0],
                             ])
    else:
        x0 = (torch.rand(4*n_agents, 1)-0.5)*10
        xbar = (torch.rand(4*n_agents, 1)-0.5)*10
    return x0, xbar