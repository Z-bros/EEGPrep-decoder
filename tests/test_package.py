import re

import pytest

import eeg_decoder
from eeg_decoder import EEGConfig


def test_import_and_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+", eeg_decoder.__version__)


def test_public_api():
    assert set(eeg_decoder.__all__) == {"EEGConfig", "__version__"}


def test_config_defaults_and_immutability():
    cfg = EEGConfig()
    assert (cfg.seed, cfg.float_dtype) == (0, "float64")
    with pytest.raises(AttributeError):
        cfg.seed = 1  # frozen


def test_config_roundtrip_and_replace():
    cfg = EEGConfig(seed=42, float_dtype="float32")
    assert EEGConfig.from_dict(cfg.to_dict()) == cfg
    new = cfg.replace(seed=7)
    assert new.seed == 7 and cfg.seed == 42  # original untouched


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"seed": -1}, ValueError),
        ({"seed": 1.5}, TypeError),
        ({"seed": True}, TypeError),
        ({"float_dtype": "float16"}, ValueError),
    ],
)
def test_config_rejects_invalid(kwargs, exc):
    with pytest.raises(exc):
        EEGConfig(**kwargs)


def test_config_from_dict_rejects_unknown_keys():
    with pytest.raises(TypeError):
        EEGConfig.from_dict({"seed": 0, "sfreq": 250})
