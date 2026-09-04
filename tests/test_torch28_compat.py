from __future__ import annotations

import warnings
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _project_metadata() -> dict[str, object]:
    import tomllib

    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_package_declares_torch_27_plus_compatibility_release() -> None:
    metadata = _project_metadata()["project"]

    assert metadata["version"] == "1.6.1+simtoolreal.2a991753.compat3"
    assert "torch>=2.7,<3" in metadata["dependencies"]


def test_training_code_uses_torch_amp_namespace() -> None:
    files = (
        ROOT / "rl_games/algos_torch/a2c_continuous.py",
        ROOT / "rl_games/algos_torch/a2c_discrete.py",
        ROOT / "rl_games/common/a2c_common.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)

    assert "torch.cuda.amp.autocast" not in source
    assert "torch.cuda.amp.GradScaler" not in source
    assert "torch.amp.autocast" in source
    assert "torch.amp.GradScaler" in source


@pytest.mark.skipif(not __import__("torch").__version__.startswith("2.8."), reason="requires Torch 2.8")
def test_torch28_amp_api_supports_training_step() -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(4, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = model(torch.randn(8, 4, device=device)).square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    assert torch.isfinite(loss).item()
    assert not [warning for warning in caught if issubclass(warning.category, FutureWarning)]
