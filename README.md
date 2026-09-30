# cvorat — byte-level language-model architecture experiments

**What:** A personal notebook log (71 notebooks + two shared layer libraries)
exploring small byte-level LMs: grouped/windowed attention, dynamic convolution,
cumulative-state layers (`CumAvg` / `CumMax` / `CumDecay` with a hand-written
Triton kernel), LoRA-style factorisation and RoPE variants.

**Why it exists:** Scratch research. The `nn.py` / `nn2.py` pair is a compact
custom layer library; each notebook instantiates a model variant with different
vocab size / width / window / grouping hyperparameters and runs a short
next-byte-prediction training pass to compare. No results were ever written up —
this is a lab notebook, not a product.

**How to run:** no single entrypoint. Requires CUDA + a GPU.
1. Provide a byte-level corpus, e.g. `py65536.bin`, one level up from the
   notebooks (see `Dl` in `nn2.py`; the corpus used here is **not** included).
2. `pip install torch numpy matplotlib triton flash-attn liger-kernel`
3. Open a notebook (start with `12.ipynb`) and run top to bottom.

Smoke test of the reusable library only:
```
python -c "import torch, nn2; print(nn2.SwiGLU(64, 128)(torch.randn(2, 8, 64)).shape)"
```

**Status:** current (2025–2026 era) — PyTorch 2.x, Triton, flash-attn,
Liger-Kernel — but unmaintained personal scratch: no README, no pinned
environment, no saved results, duplicated layer definitions across notebooks.
Kept and published as an honest experiment archive; expect to read code, not
docs.

**License:** none present. As the owner's own work, add an explicit licence
(e.g. MIT) before publishing; until then, all rights reserved.
