import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm2d) :
            module.eval()


def initialize_linear(model):
    if type(model) == nn.Linear:
        nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)


def initialize_3d(model):
    if type(model) == nn.Conv3d:
        nn.init.xavier_uniform_(model.weight)
        if model.bias != None:
            model.bias.data.fill_(0.01)


    if type(model) == nn.BatchNorm3d:
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)


# ===========================
# Utils for Channel Exchange 
# ===========================
class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]


# ===================
# Tensor Operation
# ===================
def stack_first_dim(x):
    """
    Method to combine the first two dimension of an array
    """
    x_shape = x.size()
    new_shape = [x_shape[0] * x_shape[1]]
    if len(x_shape) > 2:
        new_shape += x_shape[2:]
    return x.view(new_shape)


def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)


# ===============
# Basic Layers
# ===============
def linear_classifier(x, param_dict):
    """
    Classifier.
    """
    return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])


def average_pool(x, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)):
    return F.avg_pool2d(x, kernel_size=kernel_size, padding=padding, stride=stride)