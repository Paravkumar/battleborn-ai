from __future__ import annotations

import importlib.util
from pathlib import Path

from torch import nn


def load_merge_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "merge_lora.py"
    spec = importlib.util.spec_from_file_location("merge_lora", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_tied_weight_keys_converts_lists_to_dicts() -> None:
    merge_lora = load_merge_module()

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._tied_weights_keys = ["lm_head.weight"]
            self.child = nn.Linear(2, 2, bias=False)
            self.child._tied_weights_keys = ("child.weight",)

    model = DummyModel()

    merge_lora.normalize_tied_weight_keys(model)

    assert model._tied_weights_keys == {"lm_head.weight": "lm_head.weight"}
    assert model.child._tied_weights_keys == {"child.weight": "child.weight"}
