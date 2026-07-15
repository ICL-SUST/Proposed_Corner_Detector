"""SOGDD 核构造与多方向滤波模块。"""

from __future__ import annotations

import numpy as np
from scipy import signal

from utils import DetectorConfig


def build_sogdd_kernels(config: DetectorConfig) -> np.ndarray:
    """构造 K 个方向的二阶高斯方向导数（SOGDD）卷积核。"""

    config.validate()
    radius = config.effective_kernel_radius
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y = np.meshgrid(coords, coords, indexing="ij")
    variance = config.gaussian_variance

    kernels = np.empty(
        (
            config.num_directions,
            2 * radius + 1,
            2 * radius + 1,
        ),
        dtype=np.float64,
    )

    for k in range(config.num_directions):
        theta = k * np.pi / config.num_directions
        xr = x * np.cos(theta) + y * np.sin(theta)
        yr = -x * np.sin(theta) + y * np.cos(theta)

        gaussian = (1.0 / (2.0 * np.pi * variance)) * np.exp(
            -(xr**2 * config.rho + yr**2 / config.rho)
            / (2.0 * variance)
        )
        derivative = (config.rho / variance) * (
            (config.rho / variance) * xr**2 - 1.0
        )
        kernel = gaussian * derivative

        # 消除离散截断引起的非零直流分量。
        kernels[k] = kernel - np.mean(kernel)

    return kernels


def build_neighbourhood_mask(config: DetectorConfig) -> np.ndarray:
    """构造局部 Gram/自相关矩阵使用的圆盘或方形邻域。"""

    radius = int(config.neighbourhood_radius)

    if config.neighbourhood_shape == "square":
        return np.ones(
            (2 * radius + 1, 2 * radius + 1),
            dtype=np.float64,
        )

    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    row, col = np.meshgrid(coords, coords, indexing="ij")

    # +1 与原始实现的离散圆盘定义保持一致。
    return (
        (row**2 + col**2) <= (radius**2 + 1.0)
    ).astype(np.float64)


def filter_image(
    gray: np.ndarray,
    kernels: np.ndarray,
    padding: int,
) -> np.ndarray:
    """对灰度图进行多方向 SOGDD 卷积。"""

    padded = np.pad(
        gray,
        ((padding, padding), (padding, padding)),
        mode="symmetric",
    )

    responses = np.empty(
        (kernels.shape[0], *padded.shape),
        dtype=np.float64,
    )

    for k, kernel in enumerate(kernels):
        responses[k] = signal.convolve2d(
            padded,
            kernel,
            mode="same",
            boundary="fill",
            fillvalue=0.0,
        )

    return responses
