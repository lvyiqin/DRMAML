import pdb

import numpy as np
import torch
from scipy.stats import gaussian_kde
from scipy.optimize import brentq

def drloss(batch_loss, confidence_level):
    # estimate the VaR_alpha according to kernel density estimator
    kde = gaussian_kde(batch_loss.detach().cpu().numpy())
    try:
        target_func = lambda x: kde.integrate_box_1d(-np.inf, x) - confidence_level
        VaR_alpha = brentq(target_func, np.min(batch_loss.detach().cpu().numpy()), np.max(batch_loss.detach().cpu().numpy()))
    except ValueError:
        x = np.linspace(np.min(batch_loss.detach().cpu().numpy()), np.max(batch_loss.detach().cpu().numpy()), 1000)
        pdf = kde.evaluate(x)
        cdf = np.cumsum(pdf) / np.sum(pdf)
        index = np.argmax(cdf >= confidence_level)
        VaR_alpha = x[index]
    # pdb.set_trace()
    # calculate the meta loss
    tail_loss = [i - VaR_alpha if (i - VaR_alpha) > 0 else torch.tensor(0.).cuda() for i in batch_loss]
    # id, tail_loss = [i - VaR_alpha if (i - VaR_alpha) > 0 else torch.tensor(0.).cuda() for i in enumerate(batch_loss)]
    tail_loss = []
    id = []
    for i, tloss in enumerate(batch_loss):
        if (tloss - VaR_alpha) > 0:
            tail_loss.append(tloss - VaR_alpha)
            id.append(i)
        else:
            tail_loss.append(torch.tensor(0.).cuda())
    # pdb.set_trace()
    new_batch_loss = torch.stack(tail_loss).mean()
    id = torch.tensor(id).cuda()
    # factor = 1 / (1 - confidence_level)
    # loss_meta = VaR_alpha + factor * new_batch_loss
    # return loss_meta
    return new_batch_loss, id


def calcu_cvar(data):
    alpha = 0.5
    max_data = sorted(data, reverse=False)[0:int(len(data) * (1-alpha))]
    cvar=np.mean(max_data)
    return cvar