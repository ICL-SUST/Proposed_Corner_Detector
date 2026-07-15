"""局部自相关矩阵与角点响应计算模块。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import signal

from sogdd import (
    build_neighbourhood_mask,
    build_sogdd_kernels,
    filter_image,
)
from utils import ArrayLikeImage, DetectorConfig, to_grayscale_float64


@dataclass
class ResponseResult:
    """角点响应计算结果。"""

    response: np.ndarray
    valid_mask: np.ndarray
    elapsed_seconds: float


def compute_corner_measure(
    directional: np.ndarray,
    image_shape: tuple[int, int],
    padding: int,
    neighbourhood_mask: np.ndarray,
    neighbourhood_radius: int,
    block_rows: int,
) -> np.ndarray:
    """计算论文中的核心角点度量。

    对每个像素，以多方向 SOGDD 绝对响应构造局部 Gram/自相关矩阵 M：

        response = det(M) / (trace(M) + eps)
    """

    height, width = image_shape
    num_directions = directional.shape[0]
    radius = int(neighbourhood_radius)
    absolute = np.abs(directional)

    measure = np.empty((height, width), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    col_start = padding - radius
    col_stop = padding + width + radius

    for row_start in range(0, height, block_rows):
        row_stop = min(height, row_start + block_rows)
        block_height = row_stop - row_start

        src_row_start = padding + row_start - radius
        src_row_stop = padding + row_stop + radius

        gram = np.empty(
            (
                block_height,
                width,
                num_directions,
                num_directions,
            ),
            dtype=np.float64,
        )

        for i in range(num_directions):
            first = absolute[
                i,
                src_row_start:src_row_stop,
                col_start:col_stop,
            ]

            for j in range(i, num_directions):
                second = absolute[
                    j,
                    src_row_start:src_row_stop,
                    col_start:col_stop,
                ]

                products = signal.convolve2d(
                    first * second,
                    neighbourhood_mask,
                    mode="valid",
                )
                gram[:, :, i, j] = products
                gram[:, :, j, i] = products

        trace = np.trace(gram, axis1=-2, axis2=-1)
        determinant = np.linalg.det(gram)
        block_measure = determinant / (trace + eps)

        # Gram 行列式理论上非负，仅修正数值舍入导致的微小负值。
        scale = max(1.0, float(np.nanmax(np.abs(determinant))))
        tolerance = np.finfo(np.float64).eps * scale
        tiny_negative = (
            (determinant < 0.0)
            & (determinant >= -tolerance)
        )
        block_measure[tiny_negative] = 0.0
        block_measure[~np.isfinite(block_measure)] = -np.inf

        measure[row_start:row_stop] = block_measure

    return measure


def build_valid_mask(
    shape: tuple[int, int],
    margin: int,
) -> np.ndarray:
    """排除卷积与邻域不可靠的图像边缘。"""

    height, width = shape
    mask = np.ones(shape, dtype=bool)

    if margin <= 0:
        return mask

    if 2 * margin >= min(height, width):
        raise ValueError(
            f"suppression margin {margin} "
            f"is too large for image shape {shape}"
        )

    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False
    return mask


def compute_response(
    image: ArrayLikeImage,
    config: DetectorConfig,
) -> ResponseResult:
    """完成灰度转换、SOGDD 滤波和角点响应计算。"""

    config.validate()
    start = time.perf_counter()

    gray = to_grayscale_float64(image)
    kernels = build_sogdd_kernels(config)
    directional = filter_image(
        gray,
        kernels,
        config.padding_width,
    )

    response = compute_corner_measure(
        directional=directional,
        image_shape=gray.shape,
        padding=config.padding_width,
        neighbourhood_mask=build_neighbourhood_mask(config),
        neighbourhood_radius=config.neighbourhood_radius,
        block_rows=config.block_rows,
    )

    valid_mask = build_valid_mask(
        gray.shape,
        config.suppression_margin,
    )

    return ResponseResult(
        response=response,
        valid_mask=valid_mask,
        elapsed_seconds=time.perf_counter() - start,
    )
