"""Minimal CLI example for scoring one image pair with RFFM.

Usage:
	python example.py --pred path/to/prediction.png --target path/to/reference.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from RFFM.rffm import RFFM


def load_bgr_image(image_path: Path):
	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	return image


def main() -> int:
	parser = argparse.ArgumentParser(description="Compute RFFM scores for an image pair.")
	parser.add_argument("--pred", required=True, type=Path, help="Path to the predicted/restored image")
	parser.add_argument("--target", required=True, type=Path, help="Path to the reference image")
	args = parser.parse_args()

	pred = load_bgr_image(args.pred)
	target = load_bgr_image(args.target)

	if pred.shape != target.shape:
		raise ValueError(
			f"Input images must have the same shape, got {pred.shape} and {target.shape}."
		)

	rffm = RFFM()
	scores = rffm.score(pred, target)

	for key, value in scores.to_dict().items():
		print(f"{key}: {value:.6f}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
