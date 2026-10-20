"""Imports from BrainCapture's `eegprep` package (June 2025 layout), used for transformer preprocessing and fine-tuning.

`eegprep` is not yet publicly released. The biomarker, random forest, concept, and TCAV stages do not need it.
"""
try:
    from dataset.h5py_dataset import H5PYDatasetLabeled
    from dataset.transforms import StandardizeLabel
except ImportError as e:
    raise ImportError(
        "Fine-tuning requires BrainCapture's `eegprep` package (v0.1.x layout, providing `dataset.h5py_dataset` and "
        "`dataset.transforms`), which is not yet public. See the README section 'Dependencies on eegprep'."
    ) from e
