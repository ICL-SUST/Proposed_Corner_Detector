"""通用配置、图像读取与可视化工具。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from PIL import Image, ImageDraw


ArrayLikeImage = Union[str, Path, np.ndarray, Image.Image]
ThresholdMode = Literal["relative", "quantile", "absolute", "none"]
NeighbourhoodShape = Literal["disk", "square"]


@dataclass(frozen=True)
class DetectorConfig:
    """所提 SOGDD 角点检测器的参数配置。

    默认值为实验中冻结使用的配置：
    sigma=1.2, K=4, 邻域半径=2, NMS 半径=1,
    正响应 0.99 分位数阈值。
    """

    sigma: float = 1.2
    num_directions: int = 4
    rho: float = 1.0
    kernel_radius: Optional[int] = None

    neighbourhood_radius: int = 2
    neighbourhood_shape: NeighbourhoodShape = "disk"

    nms_radius: float = 1.0
    threshold_mode: ThresholdMode = "quantile"
    threshold_value: float = 0.99

    border_extra: int = 2
    block_rows: int = 48
    unique_local_maxima: bool = False
    max_keypoints: Optional[int] = None

    def validate(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.num_directions < 2:
            raise ValueError("num_directions must be at least 2")
        if self.rho <= 0:
            raise ValueError("rho must be positive")
        if self.effective_kernel_radius < 1:
            raise ValueError("kernel radius must be positive")
        if self.neighbourhood_radius < 1:
            raise ValueError("neighbourhood_radius must be positive")
        if self.neighbourhood_shape not in {"disk", "square"}:
            raise ValueError("unsupported neighbourhood_shape")
        if self.nms_radius <= 0:
            raise ValueError("nms_radius must be positive")
        if self.threshold_mode not in {"relative", "quantile", "absolute", "none"}:
            raise ValueError("unsupported threshold_mode")
        if self.threshold_mode in {"relative", "quantile"} and not (
            0 <= self.threshold_value <= 1
        ):
            raise ValueError(
                f"{self.threshold_mode} threshold_value must be in [0, 1]"
            )
        if self.threshold_mode == "absolute" and not np.isfinite(
            self.threshold_value
        ):
            raise ValueError("absolute threshold_value must be finite")
        if self.border_extra < 0:
            raise ValueError("border_extra cannot be negative")
        if self.block_rows < 1:
            raise ValueError("block_rows must be positive")
        if self.max_keypoints is not None and self.max_keypoints < 1:
            raise ValueError("max_keypoints must be positive or None")

    @property
    def gaussian_variance(self) -> float:
        return float(self.sigma) ** 2

    @property
    def effective_kernel_radius(self) -> int:
        if self.kernel_radius is not None:
            return int(self.kernel_radius)
        return max(3, int(math.ceil(4.0 * float(self.sigma))))

    @property
    def padding_width(self) -> int:
        return self.effective_kernel_radius + self.neighbourhood_radius + 2

    @property
    def suppression_margin(self) -> int:
        return (
            self.effective_kernel_radius
            + self.neighbourhood_radius
            + self.border_extra
        )

    def to_dict(self) -> dict:
        values = asdict(self)
        values.update(
            {
                "gaussian_variance": self.gaussian_variance,
                "effective_kernel_radius": self.effective_kernel_radius,
                "padding_width": self.padding_width,
                "suppression_margin": self.suppression_margin,
            }
        )
        return values


def to_grayscale_float64(image: ArrayLikeImage) -> np.ndarray:
    """把路径、PIL 图像或 NumPy 数组转换为连续 float64 灰度图。"""

    if isinstance(image, (str, Path)):
        with Image.open(image) as pil_image:
            array = np.asarray(pil_image)
    elif isinstance(image, Image.Image):
        array = np.asarray(image)
    else:
        array = np.asarray(image)

    if array.ndim == 2:
        gray = array.astype(np.float64, copy=False)
    elif array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[..., :3].astype(np.float64, copy=False)
        gray = (
            0.298936021293775 * rgb[..., 0]
            + 0.587043074451121 * rgb[..., 1]
            + 0.114020904255103 * rgb[..., 2]
        )
        if np.issubdtype(array.dtype, np.integer):
            gray = np.rint(gray)
    else:
        raise ValueError(f"image must be HxW or HxWx3; got {array.shape}")

    gray = np.ascontiguousarray(gray, dtype=np.float64)
    if not np.isfinite(gray).all():
        raise ValueError("image contains NaN or infinite values")
    return gray


def gray_to_uint8(gray: np.ndarray) -> np.ndarray:
    """将灰度数组稳定映射为 uint8，供可视化使用。"""

    gray = np.asarray(gray, dtype=np.float64)
    if gray.size == 0:
        raise ValueError("empty image")

    lo = float(np.min(gray))
    hi = float(np.max(gray))

    if lo >= 0.0 and hi <= 255.0:
        return np.clip(np.rint(gray), 0, 255).astype(np.uint8)
    if hi > lo:
        return np.clip(
            np.rint((gray - lo) / (hi - lo) * 255.0),
            0,
            255,
        ).astype(np.uint8)
    return np.zeros(gray.shape, dtype=np.uint8)


def draw_keypoints(
    image: ArrayLikeImage,
    keypoints_xy: np.ndarray,
    radius: int = 2,
    color: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    """在图像上用方框绘制检测角点。坐标顺序为 (x, y)。"""

    gray = to_grayscale_float64(image)
    view = gray_to_uint8(gray)
    rgb = np.repeat(view[..., None], 3, axis=2)
    output = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(output)

    for x, y in np.asarray(keypoints_xy, dtype=np.int64).reshape(-1, 2):
        draw.rectangle(
            (
                int(x - radius),
                int(y - radius),
                int(x + radius),
                int(y + radius),
            ),
            outline=color,
            width=1,
        )
    return output
