import json
import struct
import pytest
from browser_export.preflight import inspect_header


def fixture(path, header, payload=b"\0" * 16):
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


def test_metadata_counts_text_and_vision_without_loading_tensors(tmp_path):
    path = fixture(
        tmp_path / "model.safetensors",
        {
            "model.language_model.weight": {
                "dtype": "BF16",
                "shape": [2, 2],
                "data_offsets": [0, 8],
            },
            "model.visual.weight": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [8, 16],
            },
        },
    )
    assert inspect_header(path) == {
        "textParameters": 4,
        "visionParameters": 2,
        "otherParameters": 0,
        "tensorBytes": 16,
    }


def test_truncated_tensor_data_is_not_an_asset_size_proof(tmp_path):
    path = fixture(
        tmp_path / "model.safetensors",
        {
            "model.language_model.weight": {
                "dtype": "F32",
                "shape": [8],
                "data_offsets": [0, 32],
            },
        },
    )
    with pytest.raises(ValueError, match="payload"):
        inspect_header(path)


def test_header_size_is_bounded_before_reading(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", 2**40))
    with pytest.raises(ValueError, match="header"):
        inspect_header(path)
