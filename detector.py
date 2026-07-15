"""所提 SOGDD 相邻角点检测器主接口。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from postprocess import extract_keypoints
from response import compute_response
from utils import ArrayLikeImage, DetectorConfig


@dataclass
class DetectionResult:
    """完整检测结果。"""

    keypoints_xy: np.ndarray
    scores: np.ndarray
    response: np.ndarray
    valid_mask: np.ndarray
    effective_threshold: float
    elapsed_seconds: float
    response_elapsed_seconds: float
    config: DetectorConfig

    @property
    def corner_num(self) -> int:
        return int(self.keypoints_xy.shape[0])


def proposed_detector(
    image: ArrayLikeImage,
    config: Optional[DetectorConfig] = None,
) -> DetectionResult:
    """运行论文所提的 SOGDD 相邻角点检测方法。"""

    if config is None:
        config = DetectorConfig()

    start = time.perf_counter()

    response_result = compute_response(
        image,
        config,
    )

    keypoints_xy, scores, threshold = extract_keypoints(
        response_result,
        config,
    )

    return DetectionResult(
        keypoints_xy=keypoints_xy,
        scores=scores,
        response=response_result.response,
        valid_mask=response_result.valid_mask,
        effective_threshold=threshold,
        elapsed_seconds=time.perf_counter() - start,
        response_elapsed_seconds=response_result.elapsed_seconds,
        config=config,
    )


__all__ = [
    "DetectorConfig",
    "DetectionResult",
    "proposed_detector",
]
