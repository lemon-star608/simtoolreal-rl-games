# SimToolReal RL-Games SAPG Runtime

This repository packages the native RL-Games runtime used by UniLab's optional
SimToolReal SAPG training entrypoint.

It preserves 72 Python files selected from the SimToolReal RL-Games fork. Seven
reviewed compatibility patches adapt the native runner to the Gymnasium and
NumPy versions used by UniLab. `PATCHES.md` documents every patch and
`source_manifest.json` records the upstream provenance and hashes.

The vendor intentionally does not include the 122 YAML files in the selected Source parent tree,
Source examples, notebooks, or the Source top-level `rl_games/tests/` test suite. The
runtime selection does retain test-named Python modules inside `rl_games/rl_games`, including
`common/test_utils.py`, `envs/test/**`, and `envs/test_network.py`; these are part of the 72
selected runtime identities.

## Installation

```bash
uv sync --extra mujoco --extra rlgames-sapg
```

UniLab pins the package to an immutable Git commit. Do not format files below
`rl_games/`; the selected Source files are intentionally kept byte-stable.
