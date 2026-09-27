# 0007: GPU environment target (torch / CUDA / PyTorch Geometric on Windows)

- Status: PROPOSED, unchanged in the Phase 0 revision pass. **NOT YET EVALUATED — requires verification on the local GPU machine before
  Milestone 3+** (in practice torch is first needed at Milestone 5; see below).
- Target machine: Windows laptop, RTX 5070 Ti (12 GB), 64 GB RAM, Python 3.11, "CUDA 12.8".

## What was checked in this session (cloud sandbox, no GPU)

Queried the PyPI JSON API (the PyTorch wheel index `download.pytorch.org` and `data.pyg.org` both
returned **HTTP 403 from the sandbox proxy**, so CUDA-specific Windows wheels could not be listed):

```
torch latest: 2.14.0   requires_python: >=3.10
2.12.0 ['torch-2.12.0-cp311-cp311-win_amd64.whl'] 2026-05-13
2.12.1 ['torch-2.12.1-cp311-cp311-win_amd64.whl'] 2026-06-17
2.13.0 ['torch-2.13.0-cp311-cp311-win_amd64.whl'] 2026-07-08
2.14.0 ['torch-2.14.0-cp311-cp311-win_amd64.whl'] 2026-09-02
pyg latest: 2.8.0.post1   requires_python: >=3.10   (pure-Python wheel, torch not in requires_dist)
```

CUDA runtime pinned by the **Linux** PyPI builds (read from `requires_dist`):

```
2.9.1   nvidia-cuda-runtime-cu12==12.8.90, nvidia-cudnn-cu12==9.10.2.21
2.12.1  cuda-toolkit[...]==13.0.2, nvidia-cudnn-cu13==9.20.0.48
2.14.0  cuda-toolkit[...]==13.0.3, nvidia-cudnn-cu13==9.24.0.43
```

So the default CUDA line moved from 12.8 to 13.0 somewhere between 2.9.1 and 2.12.1. Whether
`+cu128` Windows wheels are still published for 2.12 to 2.14 could not be verified here.

## Background (prior knowledge, not verified in this session; treat as uncertain)

- The RTX 50 series (Blackwell, compute capability 12.0) needs PyTorch binaries built with
  CUDA >= 12.8; torch 2.7 was the first release with cu128 wheels.
- The Windows wheels on PyPI have historically been CPU-only; CUDA builds come from the PyTorch index.
- A PyTorch CUDA wheel bundles its CUDA runtime; what the laptop must provide is an NVIDIA **driver**
  new enough for that runtime. "CUDA 12.8" on the laptop most likely refers to the driver's maximum
  supported version shown by `nvidia-smi`, or to an installed toolkit; they are different things.
  cu13x wheels need a newer driver branch than cu128 wheels.
- PyG 2.3+ works for most layers without the optional compiled extensions (`torch-scatter`,
  `torch-sparse`, `torch-cluster`, `pyg-lib`), but neighbor sampling (`NeighborLoader`, temporal
  sampling) has historically required `pyg-lib` or `torch-sparse`.

## Intended target

**Primary (conservative, matches "CUDA 12.8"):**

- Python 3.11
- `torch==2.9.1+cu128` from `https://download.pytorch.org/whl/cu128` (pinned through a uv index in
  `pyproject.toml`, only for the GPU extra)
- `torch_geometric==2.7.0` (pure Python)
- **No** `pyg-lib`, `torch-scatter`, `torch-sparse`, `torch-cluster` unless verified to install on
  Windows for that exact torch build. Neighbor sampling via the in-house as-of sampler (decision 0006 b).

Reasoning: 2.9.1 is the newest release verified here whose default CUDA line is 12.8, which makes a
cu128 build very likely to be a first-class variant; it avoids requiring a driver upgrade; the pure
Python PyG wheel avoids Windows compiled-extension risk.

**Alternative (newer):** if `nvidia-smi` shows a driver that supports CUDA 13.0, target the newest
torch with a cu130 Windows wheel (2.14.0 at the time of writing) and the newest PyG. Prefer this if
the driver allows it, since the project will live for many months.

**CI / sandbox:** CPU-only torch from PyPI for unit tests of model code (later milestones).

## Verification steps to run on the laptop (before Milestone 5)

1. `nvidia-smi` (record driver version and "CUDA Version").
2. `uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128` in a scratch venv.
3. `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"`
   Expect capability `(12, 0)` and `sm_120` in the arch list.
4. A small matmul and a backward pass on the GPU; a 12 GB memory allocation smoke test.
5. `pip install torch_geometric==2.7.0`; run a GraphSAGE forward/backward on a random graph on GPU.
6. Record everything in `docs/ENVIRONMENT.md` with real outputs.

## Why this does not block Milestones 1 to 4

Milestones 1 (generator), 2 (features), 3 (XGBoost/LightGBM) and 4 (graph construction, which can use
numpy/scipy/NetworkX) need no torch. Keeping torch out of the lockfile until Milestone 5 removes the
environment risk from the critical path.
