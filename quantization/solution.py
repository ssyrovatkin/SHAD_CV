from typing import List, Tuple

import numpy as np
import torch


# Task 1 (1 point)
class QuantizationParameters:
    def __init__(
        self,
        scale: np.float64,
        zero_point: np.int32,
        q_min: np.int32,
        q_max: np.int32,
    ):
        self.scale = scale
        self.zero_point = zero_point
        self.q_min = q_min
        self.q_max = q_max

    def __repr__(self):
        return f"scale: {self.scale}, zero_point: {self.zero_point}"


def compute_quantization_params(
    r_min: np.float32,
    r_max: np.float32,
    q_min: np.int32,
    q_max: np.int32,
) -> QuantizationParameters:
    # your code goes here \/
    scale = ((r_max - r_min) / (q_max - q_min)).astype(np.float64)
    zero_point = np.round((r_max * q_min - r_min * q_max) / (r_max - r_min)).astype(np.int32)
    q_params = QuantizationParameters(scale, zero_point, q_min, q_max)
    return q_params
    # your code goes here /\


# Task 2 (0.5 + 0.5 = 1 point)
def quantize(r: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    S = qp.scale
    Z = qp.zero_point
    q_min = qp.q_min
    q_max = qp.q_max
    q = np.clip(np.round(r / S + Z), q_min, q_max).astype(np.int8)
    return q
    # your code goes here /\


def dequantize(q: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    S = qp.scale
    Z = qp.zero_point
    r = S * (q.astype(np.float32) - Z.astype(np.float32))
    return r
    # your code goes here /\


# Task 3 (1 point)
class MinMaxObserver:
    def __init__(self):
        self.min = np.finfo(np.float32).max
        self.max = np.finfo(np.float32).min

    def __call__(self, x: torch.Tensor):
        # your code goes here \/
        curr_min = np.array(torch.min(x).item()).astype(np.float32)
        curr_max = np.array(torch.max(x).item()).astype(np.float32)
        self.min = np.minimum(self.min, curr_min)
        self.max = np.maximum(self.max, curr_max)
        # your code goes here /\


# Task 4 (1 + 1 = 2 points)
def quantize_weights_per_tensor(
    weights: np.ndarray,
) -> Tuple[np.array, QuantizationParameters]:
    # your code goes here \/
    r_min, r_max = np.min(weights), np.max(weights)
    if np.abs(r_min) > np.abs(r_max):
        r_max = np.abs(r_min)
    elif np.abs(r_min) < np.abs(r_max):
        r_min = -np.abs(r_max)

    qp = compute_quantization_params(r_min, r_max, -127, 127)
    q_weights = quantize(weights, qp)

    return q_weights, qp
    # your code goes here /\


def quantize_weights_per_channel(
    weights: np.ndarray,
) -> Tuple[np.array, List[QuantizationParameters]]:
    # your code goes here \/
    C = weights.shape[0]
    q_weights_per_channel = []
    qp_per_channel = []

    for i in range(C):
        q_weights, qp = quantize_weights_per_tensor(weights[i])
        q_weights_per_channel.append(q_weights)
        qp_per_channel.append(qp)
    q_weights_per_channel = np.stack(q_weights_per_channel, axis=0)

    return q_weights_per_channel, qp_per_channel
    # your code goes here /\


# Task 5 (1 point)
def quantize_bias(
    bias: np.float32,
    scale_w: np.float64,
    scale_x: np.float64,
) -> np.int32:
    # your code goes here \/
    q = np.clip(np.round(bias / (scale_w * scale_x)), -2147483648, 2147483647).astype(np.int32)
    return q
    # your code goes here /\


# Task 6 (2 points)
def quantize_multiplier(m: np.float64) -> [np.int32, np.int32]:
    # your code goes here \/
    n = np.ceil(-np.log2(m * 2)).astype(np.int32)
    m0 = m * np.power(2, n, dtype=np.float64)
    return n, np.round(m0 * 2 ** 31).astype(np.int32)
    # your code goes here /\


# Task 7 (2 points)
def multiply_by_quantized_multiplier(
    accum: np.int32,
    n: np.int32,
    m0: np.int32,
) -> np.int32:
    # your code goes here \/
    q = np.multiply(accum, m0, dtype=np.int64)
    bin_q = np.binary_repr(q)
    if bin_q[0] == '-':
        is_minus = True
        bin_q = bin_q[1:]
    else:
        is_minus = False

    bin_q = bin_q.zfill(64)
    point_pos = 33 - n
    first_after_point = bin_q[point_pos]
    q = np.array(int(bin_q[:point_pos], 2)).astype(np.int64)

    if first_after_point == '1':
       q += 1
    if is_minus:
        q *= -1

    return np.int32(q)
    # your code goes here /\
