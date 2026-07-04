import numpy as np
import torch

MU = 100
EPSILON = 1 / 64


def range_compressor(x, mu=MU):
    return np.log1p(mu * x) / np.log1p(mu)


def inverse_range_compressor(y, mu=MU):
    return (np.power(1 + mu, y) - 1) / mu


def range_compressor_tensor(x, mu=MU):
    return torch.log1p(mu * x) / np.log1p(mu)


def inverse_range_compressor_tensor(y, mu=MU):
    return (torch.pow(1 + mu, y) - 1) / mu


def calculate_gm(hdr, ldr):
    if torch.is_tensor(hdr) and torch.is_tensor(ldr):
        return range_compressor_tensor(hdr / (ldr + EPSILON))
    return range_compressor(hdr / (ldr + EPSILON))


def recover_hdr_from_gm(ldr, gm_compressed):
    if torch.is_tensor(ldr) and torch.is_tensor(gm_compressed):
        return (ldr + EPSILON) * inverse_range_compressor_tensor(gm_compressed.float())
    return ((ldr + EPSILON) * inverse_range_compressor(gm_compressed.astype(np.float32))).astype(np.float32)
