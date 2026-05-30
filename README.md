# RFFM

RFFM is a reference-based image quality metric for restored images. It compares a predicted image against a target image and returns an overall score in the range `[0, 1]` plus four sub-scores:

- `rime_sharpness`
- `rime_artifacts`
- `rime_texture`
- `rime_intensity`

The implementation in this repository expects a single image pair at a time and works with BGR images such as the arrays returned by `cv2.imread()`.

## Installation

Create a virtual environment, install the dependencies, then run the example script:

```bash
pip install -r requirements.txt
```

## Example

The quickest way to use RFFM is from Python:

```python
from rime import RFFM

rime = RFFM()
scores = rime.score(pred, target)

print(scores.to_dict())
print(scores.to_array())
```

Where `pred` and `target` are NumPy arrays with shape `(H, W, 3)` and BGR channel order.

### Command-line example

This repository also includes a small CLI demo in `example.py`:

```bash
python example.py --pred path/to/prediction.png --target path/to/reference.png
```

Example output:

```text
RFFM scores
	rime: 0.8231
	rime_sharpness: 0.7812
	rime_artifacts: 0.8664
	rime_texture: 0.8097
	rime_intensity: 0.8349
```

## Notes

- `config.json` contains the feature names, weights, and reference values used by the metric.
- The metric relies on PyRadiomics features extracted from the input image pair.
- If you load images with OpenCV, you can pass them directly to `RFFM.score()`.
