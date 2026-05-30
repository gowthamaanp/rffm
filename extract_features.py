import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
import radiomics
from radiomics import featureextractor
radiomics.setVerbosity(40)


def rgb_to_y_channel(img_bgr):
    """Convert BGR image to Y channel (ITU-R BT.601).
    Returns uint8 Y-channel image.
    """
    img_rgb = img_bgr[:, :, ::-1].astype(np.float64)
    y = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
    return np.clip(y, 0, 255).astype(np.uint8)


def create_extractor(features=None):
    """Create PyRadiomics feature extractor with specified features.

    Args:
        features: Optional list of specific radiomics feature names to extract,
            e.g. ["original_glcm_SumSquares", "wavelet-HL_gldm_LargeDependenceHighGrayLevelEmphasis"].
            Names must follow the PyRadiomics convention:
                {image_type}_{feature_class}_{feature_name}
            If None, all features from the enabled feature classes are extracted.

            Options: original, LoG, wavelet, gradient, square, squareroot,
                     logarithm, exponential, lbp-2D

        features: Optional list of specific radiomics feature names to extract,
            e.g. ["original_glcm_SumSquares", "wavelet-HL_gldm_LargeDependenceHighGrayLevelEmphasis"].
            Names must follow the PyRadiomics convention:
                {image_type}_{feature_class}_{feature_name}
            If None, all features from the enabled feature classes are extracted.

    Returns:
        Configured featureextractor.RadiomicsFeatureExtractor
    """
    # --- Image type name → canonical PyRadiomics name -------------------
    _CANONICAL = {
        'original':    'Original',
        'wavelet':     'Wavelet',
        'squareroot':  'SquareRoot',
        'logarithm':   'Logarithm',
        'gradient':    'Gradient',
        'square':      'Square',
        'log':         'LoG',
        'exponential': 'Exponential',
        'lbp':         'LBP2D',
        'lbp-2d':      'LBP2D',
        'lbp2d':       'LBP2D',
    }

    settings = {
        'binWidth': 25,
        'normalize': True,
        'normalizeScale': 100,
        'resampledPixelSpacing': None,  # no resampling for 2D natural images
        'interpolator': sitk.sitkBSpline,
        'removeOutliers': None,
        'force2D': True,
        'force2Ddimension': 0,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    parsed_image_types = set()   # canonical names derived from feature list
    class_feature_map  = {}      # {feature_class: [feature_name, ...]}
    if features is not None:
        for feat in features:
            parts = feat.split('_', 2)
            if len(parts) != 3:
                raise ValueError(
                    f"Cannot parse feature name '{feat}'. "
                    "Expected format: {{image_type}}_{{feature_class}}_{{feature_name}}"
                )
            img_type_raw, feat_class, feat_name = parts
            img_type_key = img_type_raw.lower().split('-')[0]
            canonical = _CANONICAL.get(img_type_key)
            if canonical is None:
                raise ValueError(
                    f"Unrecognised image type '{img_type_raw}' in feature '{feat}'."
                )
            parsed_image_types.add(canonical)
            class_feature_map.setdefault(feat_class, []).append(feat_name)
    active_types = list(_CANONICAL.values())  # de-duplicated below via set
    active_types = list(dict.fromkeys(active_types))  # preserve order, drop dups
    for canonical in active_types:
        if canonical == 'LoG':
            extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0, 5.0]})
        else:
            extractor.enableImageTypeByName(canonical)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(**class_feature_map)
    return extractor


def extract_features_from_image(img_y, extractor):
    """Extract radiomics features from a Y-channel image.
    
    Args:
        img_y: 2D numpy array (uint8), Y-channel
        extractor: RadiomicsFeatureExtractor
    
    Returns:
        OrderedDict of {feature_name: feature_value}
    """
    # Convert to SimpleITK image (3D with z=1 for force2D)
    img_3d = img_y[np.newaxis, :, :].astype(np.float64)
    sitk_img = sitk.GetImageFromArray(img_3d)
    
    # Create mask: label=1 for ROI, with a 1-pixel border of 0s
    # PyRadiomics requires both 0 and 1 in the mask (background + foreground)
    h, w = img_y.shape
    mask_2d = np.ones((h, w), dtype=np.int32)
    mask_2d[0, :] = 0  # top row = background
    mask_2d[-1, :] = 0  # bottom row = background
    mask_2d[:, 0] = 0  # left col = background
    mask_2d[:, -1] = 0  # right col = background
    mask_3d = mask_2d[np.newaxis, :, :]
    sitk_mask = sitk.GetImageFromArray(mask_3d)
    sitk_mask.CopyInformation(sitk_img)
    
    # Extract features
    result = extractor.execute(sitk_img, sitk_mask, label=1)
    
    # Filter out diagnostic features (keep only radiomics features)
    features = OrderedDict()
    for key, val in result.items():
        if not key.startswith('diagnostics_'):
            # Convert numpy types to float
            try:
                features[key] = float(val)
            except (ValueError, TypeError):
                features[key] = val
    
    return features

