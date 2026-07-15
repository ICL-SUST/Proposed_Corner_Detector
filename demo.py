"""单张图像测试脚本。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from detector import DetectorConfig, proposed_detector
from utils import draw_keypoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the proposed SOGDD adjacent-corner detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        type=Path,
        help="输入图像路径，例如 Table.png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("proposed_output"),
        help="输出目录",
    )
    parser.add_argument(
        "--save-response",
        action="store_true",
        help="保存 response.npy",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["relative", "quantile", "absolute", "none"],
        default="quantile",
    )
    parser.add_argument(
        "--threshold-value",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=None,
    )
    return parser


def save_outputs(
    image_path: Path,
    result,
    output_dir: Path,
    save_response: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "keypoints.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "score"])

        for (x, y), score in zip(
            result.keypoints_xy,
            result.scores,
        ):
            writer.writerow(
                [int(x), int(y), f"{float(score):.17g}"]
            )

    draw_keypoints(
        image_path,
        result.keypoints_xy,
    ).save(output_dir / "keypoints_overlay.png")

    np.save(
        output_dir / "valid_mask.npy",
        result.valid_mask,
    )

    if save_response:
        np.save(
            output_dir / "response.npy",
            result.response,
        )

    summary = {
        "image": str(image_path),
        "corner_num": result.corner_num,
        "effective_threshold": result.effective_threshold,
        "elapsed_seconds": result.elapsed_seconds,
        "response_elapsed_seconds": (
            result.response_elapsed_seconds
        ),
        "detector_config": result.config.to_dict(),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    config = DetectorConfig(
        threshold_mode=args.threshold_mode,
        threshold_value=args.threshold_value,
        max_keypoints=args.max_keypoints,
    )

    result = proposed_detector(
        args.image,
        config,
    )

    save_outputs(
        args.image,
        result,
        args.output_dir,
        args.save_response,
    )

    print(
        json.dumps(
            {
                "corner_num": result.corner_num,
                "effective_threshold": (
                    result.effective_threshold
                ),
                "elapsed_seconds": result.elapsed_seconds,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
