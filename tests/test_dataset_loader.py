import os

import pytest

from alignit.utils.dataset import load_dataset


def test_load_dataset_local(monkeypatch, tmp_path):
    class Dummy:
        def __init__(self, path):
            self.path = path

    def fake_load_from_disk(p):
        return Dummy(p)

    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    # Patch inside module
    import alignit.utils.dataset as ds
    ds.load_from_disk = fake_load_from_disk

    d = load_dataset(str(tmp_path))
    assert isinstance(d, Dummy)
    assert d.path == str(tmp_path)


def test_load_dataset_hub(monkeypatch):
    class Dummy:
        def __init__(self, name):
            self.name = name

    def fake_hf_load_dataset(name):
        return Dummy(name)

    import alignit.utils.dataset as ds
    ds.hf_load_dataset = fake_hf_load_dataset

    d = load_dataset("my-dataset/name")
    assert isinstance(d, Dummy)
    assert d.name == "my-dataset/name"
