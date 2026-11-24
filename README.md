# ffx-kit

GPU-verified FidelityFX CAS replacements for Xbox-to-PC port projects.

## What this is
- Compiles the official AMD FidelityFX-CAS sources (MIT) with DXC to SPIR-V.
- Runs them on a real GPU through Vulkan (wgpu-py).
- Verifies every output against a CPU golden model that mirrors AMD's exact
  arithmetic: APrxLoRcpF1 / APrxMedRcpF1 / APrxLoSqrtF1 bit tricks,
  green-channel weight path, D3D saturate(NaN)=0 semantics.

No game files, no console binaries, no keys. Only official open sources.

## Why
Port projects translate console shaders, but translated output is usually
unverified. A naive "mathematically correct" reference disagrees with AMD's
fast-math paths by up to ~8.5% of signal. This kit reproduces the intended
approximations, so verification is meaningful (a few fp32 ULPs agreement).

## Results (2026-11-24)
- GPU: NVIDIA RTX 4070 Ti, Vulkan, driver 621.24
- Compiler: DXC 1.9.2609.5
- Variants: 8/8 fp32 sharpen-only (BETTER_DIAGONALS x GO_SLOWER x SLOW), all PASS
- Worst max abs diff: 4.77e-07
- 4K 3840x2160: PASS, max diff 2.98e-07, full cycle 219 ms incl. readback

See manifest.json for per-variant SHA-256 and per-test diffs.

## Reproduce
1. Install Python 3.12+, Git, DXC (winget: Git.Git, Microsoft.DirectX.ShaderCompiler)
2. pip install numpy wgpu pillow
3. git clone --recursive https://github.com/GPUOpen-Effects/FidelityFX-CAS.git
4. python verify_all.py

## Scope / next
- fp32 sharpen-only today; fp16 packed (CasFilterH) and scaling next.
- Intended use: verified PC-side replacements a port runtime can swap in for
  the corresponding FidelityFX programs. Matching to console binaries is left
  to projects that have legal access to them.

## License
MIT (this repo). AMD FidelityFX sources remain MIT, see THIRD_PARTY_NOTICES.md.
