import torch
import numpy as np

eps = 1.0e-8


def csp_loss(csp_data, soft_assignments, discount=0.95):
    # collect prob. of satisfying each constraint
    sat_prob = csp_data.constraint_sat_prob(soft_assignments)
    sat_prob = torch.cat([p for p in sat_prob.values()], dim=0)

    # min. negative log-likelihood
    loss = -torch.log(sat_prob + eps).mean(dim=0)

    # weighted sum with discount factor through time
    weights = discount ** torch.arange(loss.shape[0]-1, -1, -1, device=csp_data.device)
    loss = (weights * loss).sum()
    return loss


def mis_loss(csp_data, soft_assignments, discount=0.95, kappa=1.0):
    sat_prob = csp_data.constraint_sat_prob(soft_assignments)
    sat_prob = torch.cat([p for p in sat_prob.values()], dim=0)
    is_loss = -torch.log(sat_prob + eps).mean(dim=0)
    max_loss = soft_assignments.mean(dim=0)
    loss = (is_loss + kappa) * (max_loss + 1)

    weights = discount ** torch.arange(loss.shape[0]-1, -1, -1, device=csp_data.device)
    loss = (weights * loss).sum()
    return loss
