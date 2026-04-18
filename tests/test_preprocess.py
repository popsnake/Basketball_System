import numpy as np

from models_dl.preprocess import resample_or_pad_flat


def test_pad_shorter_sequence():
    x = np.ones((10, 75))
    y = resample_or_pad_flat(x, 31)
    assert y.shape == (31, 75)
    assert np.allclose(y[:10], 1.0)
    assert np.allclose(y[10:], 0.0)


def test_downsample_longer_sequence():
    x = np.arange(62 * 75, dtype=np.float64).reshape(62, 75)
    y = resample_or_pad_flat(x, 31)
    assert y.shape == (31, 75)
