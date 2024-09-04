#coding=utf-8
import math
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def cosine_scheduler(optimizer, lr, min_lr, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, lr, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule




def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    global lr, alpha, beta
    if epoch + 1 < 100:
        lr = 0.1
        alpha = 0.1
        beta = 0.9
    elif epoch + 1 >= 100 and epoch + 1 < 150:
        lr = 0.01
        alpha = 0.5
        beta = 0.5
    else:
        lr = 0.001
        alpha = 0.5
        beta = 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return alpha, beta

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


def IoU(A, B):
    xmin_a, ymin_a, xmax_a, ymax_a = A
    xmin_b, ymin_b, xmax_b, ymax_b = B
    weight = min(xmax_a, xmax_b) - max(xmin_a, xmin_b)
    height = min(ymax_a, ymax_b) - max(ymin_a, ymin_b)
    if weight<=0 or height<=0:
        return 0
    sa    = (xmax_a-xmin_a)*(ymax_a-ymin_a)
    sb    = (xmax_b-xmin_b)*(ymax_b-ymin_b)
    inter = weight*height
    union = sa+sb-inter
    return inter/(union+1e-12)


def gaussian(x, y, ux, uy, sx, sy, sxy, pred=None):
    c   = -1/(2*(1-sxy**2/sx/sy))
    dx  = (x-ux)**2/sx
    dy  = (y-uy)**2/sy
    dxy = (x-ux)*(y-uy)*sxy/sx/sy
    return np.exp(c*(dx-2*dxy+dy))


def compute_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)