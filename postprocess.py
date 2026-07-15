"""局部极大值抑制、阈值筛选与关键点输出。"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import ndimage

from response import ResponseResult
from utils import DetectorConfig, ThresholdMode


def matlab_disk_footprint(radius: float) -> np.ndarray:
    """生成与 MATLAB disk 邻域接近的离散 NMS 模板。"""

    half = int(math.ceil(radius))
    coords = np.arange(-half, half + 1, dtype=np.float64)
    row, col = np.meshgrid(coords, coords, indexing="ij")

    nearest_row = np.maximum(np.abs(row) - 0.5, 0.0)
    nearest_col = np.maximum(np.abs(col) - 0.5, 0.0)

    footprint = (
        nearest_row**2 + nearest_col**2
        <= radius**2 + 1e-12
    )
    footprint[half, half] = True
    return footprint


def effective_threshold(
    response: np.ndarray,
    valid_mask: np.ndarray,
    mode: ThresholdMode,
    value: float,
) -> float:
    """根据配置计算实际阈值。"""

    values = np.asarray(
        response,
        dtype=np.float64,
    )[valid_mask]
    values = values[np.isfinite(values)]
    positive = values[values > 0]

    if mode == "none":
        return -np.inf

    if mode == "absolute":
        return float(value)

    if positive.size == 0:
        return np.inf

    if mode == "relative":
        return float(value) * float(np.max(positive))

    if mode == "quantile":
        return float(np.quantile(positive, float(value)))

    raise ValueError(f"unsupported threshold mode: {mode}")


def local_maximum_candidates(
    response: np.ndarray,
    valid_mask: np.ndarray,
    nms_radius: float,
    unique_local_maxima: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """提取按响应值降序排列的局部极大值候选点。"""

    work = np.where(
        valid_mask & np.isfinite(response),
        response,
        -np.inf,
    )

    footprint = matlab_disk_footprint(nms_radius)
    footprint_size = int(np.count_nonzero(footprint))

    highest = ndimage.maximum_filter(
        work,
        footprint=footprint,
        mode="constant",
        cval=-np.inf,
    )
    local = work == highest

    if unique_local_maxima and footprint_size > 1:
        second = ndimage.rank_filter(
            work,
            rank=footprint_size - 2,
            footprint=footprint,
            mode="constant",
            cval=-np.inf,
        )
        local &= highest != second

    rows, cols = np.nonzero(
        local & valid_mask & np.isfinite(work)
    )
    scores = work[rows, cols]

    if scores.size:
        order = np.argsort(scores)[::-1]
        rows = rows[order]
        cols = cols[order]
        scores = scores[order]

    keypoints_xy = np.column_stack(
        (cols, rows)
    ).astype(np.int64, copy=False)

    return keypoints_xy, np.asarray(
        scores,
        dtype=np.float64,
    )


def filter_candidates_by_threshold(
    candidate_xy: np.ndarray,
    candidate_scores: np.ndarray,
    threshold: float,
    max_keypoints: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """按阈值及最大关键点数筛选候选。"""

    keep = np.asarray(candidate_scores) > float(threshold)
    keypoints_xy = np.asarray(candidate_xy)[keep]
    scores = np.asarray(candidate_scores)[keep]

    if max_keypoints is not None:
        keypoints_xy = keypoints_xy[:max_keypoints]
        scores = scores[:max_keypoints]

    return (
        keypoints_xy.astype(np.int64, copy=False),
        scores.astype(np.float64, copy=False),
    )


def extract_keypoints(
    response_result: ResponseResult,
    config: DetectorConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """完成 NMS、阈值筛选并返回角点坐标与响应值。"""

    candidate_xy, candidate_scores = local_maximum_candidates(
        response_result.response,
        response_result.valid_mask,
        config.nms_radius,
        config.unique_local_maxima,
    )

    threshold = effective_threshold(
        response_result.response,
        response_result.valid_mask,
        config.threshold_mode,
        config.threshold_value,
    )

    keypoints_xy, scores = filter_candidates_by_threshold(
        candidate_xy,
        candidate_scores,
        threshold,
        config.max_keypoints,
    )

    return keypoints_xy, scores, threshold
