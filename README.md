[English](README.md) · [简体中文](README.zh-CN.md)

<p align="center">
  <img src="assets/logo.svg" width="112" alt="cvorat logo">
</p>

<h1 align="center">cvorat</h1>

<p align="center"><strong>Byte-level language-model architecture experiments</strong></p>

<p align="center">
One architecture idea per notebook: windowed and grouped attention, dynamic convolution,
cumulative-state layers, LoRA factorisation and RoPE variants, built on a compact PyTorch layer kit.
</p>

---

cvorat is a lab notebook for small byte-level language models. Instead of one framework with a
configuration flag for every idea, it keeps each experiment as a standalone notebook and two small
shared libraries. The point is to compare architectural choices quickly, not to ship a training
framework.

**Status:** public experiment archive. The notebooks are the record; there is no maintained
release, no saved benchmark table and no pinned environment. See
[docs/project-status.md](docs/project-status.md) for the archive note.

## Quick start

Full training needs a CUDA GPU and the stack used by the notebooks.

```bash
git clone https://github.com/ZisIsNotZis/cvorat.git
cd cvorat
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch numpy matplotlib triton flash-attn liger-kernel ipython jupyter
```

Smoke-test the shared layer kit:

```bash
python -c "import torch; from nn2 import Swi; print(Swi(64, 8, 8)(torch.randn(2, 8, 64)).shape)"
# torch.Size([2, 8, 8])
```

Then open a notebook from the repository root, for example [`12.ipynb`](12.ipynb), and run it top
to bottom. The experiments expect a raw byte corpus: one `uint8` value per token. Corpus paths such
as `../../py65536.bin` are hard-coded in the notebooks, so point them at your own file or construct
the loader directly:

```python
from nn2 import Dl

dl = Dl(a="/path/to/py.bin")   # memmap of raw bytes
xb, yb = next(iter(dl))
```

Not every notebook needs every package: `flash-attn` is used by the attention layers and most of the
numbered notebooks, while `liger-kernel` appears in the later Liger variants.

## What it can do now

- **Byte-level data path.** `nn2.Dl` memmaps a `uint8` corpus and returns fixed-length windows, with
  random-byte corruption applied to the input and the original bytes as the next-byte target.
- **Windowed and grouped attention.** `nn.Attn` and later variants wrap FlashAttention with causal
  windows, grouped query/key/value projections and optional ALiBi slopes.
- **Convolution next to attention.** `nn2.Cnv` is the windowed, grouped conv block; the `cpa*`
  experiments stack it in the same residual block as attention and a polynomial layer, sometimes
  under a pooling hierarchy. `nn2.Dycnv` is the dynamic-weight variant in the kit: it generates its
  convolution weights per position.
- **Cumulative-state layers.** `nn2.Cumavg` and `nn2.Cummax` are causal running summaries, and the
  kit carries a signed cumulative-decay prototype with a fused Triton kernel (`cumdecay_fuseker`)
  that keeps the decay in log space.
- **Compact building blocks.** LoRA-factorised linear layers, SwiGLU, RoPE and custom remapping
  functions make it cheap to assemble a variant without a large framework.
- **A readable experiment log.** 68 notebooks cover model shapes, U-Net-style pooling, Liger fusion,
  kernel prototypes and the occasional side experiment. They are meant to be read and rerun, not
  imported as a package.

## Architecture

Two shared layers, one notebook per experiment:

```text
nn.py                  first-generation layer kit: attention, SwiGLU, RoPE, remap, plotting
nn2.py                 second-generation kit: dynamic conv, cumulative-state layers,
                       LoRA, Triton fused kernel, byte-level data loader
0.*–12*.ipynb          numbered experiment log; each notebook is self-contained
cpa*.ipynb             combined convolution / polynomial / attention blocks
scratch-*.ipynb        side experiments: Liger kernels, serving, entropy math, cumulative RoPE
docs/project-status.md archive note and boundaries
assets/logo.svg        project mark
```

The data flow is deliberately small:

```text
raw bytes → Dl window + corruption → model assembled from nn / nn2 blocks
          → next-byte cross entropy → Plot / notebook output
```

Notebooks duplicate layer definitions on purpose: each one is a snapshot of the model at that point
in the search. `nn.py` and `nn2.py` are the two import surfaces, and most notebooks pull helpers
from one of them (`from nn import …`, `from nn2 import *`) while redefining the model classes they
are studying.

## How to develop

This repository is a notebook archive, not an installed package.

- Run notebooks with Jupyter from the repository root so `import nn` / `import nn2` resolves.
  `nn2.py` imports `triton` and `IPython.display` at module load, so both must be installed even
  for blocks that never touch them.
- `nn.py` imports FlashAttention at module load. `nn2.py` imports it optionally and only needs it
  for the FlashAttention-based attention layer.
- There is no test suite, no CI, no lockfile and no build step. The smoke test above is the quickest
  check that the shared kit imports and runs.
- The notebooks hard-code widths, window sizes, vocab sizes and corpus paths. Change those values
  in the notebook you are running; do not expect a global config file.
- Reusing a block in a new experiment is the intended workflow. Copy the block into a new notebook
  first, then fold the useful part back into `nn2.py` only when it is clearly stable.
- Some blocks in `nn2.py` drifted away from the notebook copies that actually ran: `Cumdecay` and
  `cumdecay_fuse` reference helpers that no longer exist under those names (`cum`, `hr`, `normr`,
  `tl.prograc_id`). Trust the notebook, not the module, when the two disagree.

## How to contribute

This is an archive, so feature requests are out of scope. Reproduction reports and focused fixes are
welcome:

1. Open an issue naming the notebook and the environment that failed, with the traceback.
2. For a code change, keep it small and point it at `nn.py`, `nn2.py` or the notebook it repairs.
3. Do not mass-reformat or rewrite notebooks; their duplicated code is part of the record.
4. Contributions are accepted under the repository's AGPL-3.0 license.

## Roadmap

### Implemented

- [x] First-generation layer kit (`nn.py`) with windowed/grouped attention, SwiGLU, RoPE and remap.
- [x] Second-generation kit (`nn2.py`) with dynamic convolution, LoRA and cumulative-state layers.
- [x] Signed cumulative-decay prototype with a fused Triton kernel (log-space decay; the numbered
      notebooks do not use it).
- [x] Byte-level memmap loader with random-byte corruption targets.
- [x] 68 notebooks covering model shapes, pooling hierarchies, Liger variants and kernel prototypes.
- [x] AGPL-3.0 license, project mark and public-facing documentation.

### Planned

- [ ] Write up what the experiments showed. No results were recorded, so nothing is claimed yet.
- [ ] Make one run reproducible end to end: pinned environment, fixed seed, documented corpus.
- [ ] Add a small sample corpus so the smoke path does not depend on a private file.
- [ ] Consolidate the duplicated layer definitions that have settled into `nn2.py`.
- [ ] Turn the most useful model families into one runnable recipe each.

### Future vision

The long-term value is not another model implementation; it is a reproducible comparison of small
architectural ideas under the same corpus and evaluation. The notebooks can stay the lab, while the
shared kit becomes the stable surface and the results become a short, honest report. Until that
write-up exists, this repository is the evidence, not the conclusion.

## License

[GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).
