[English](README.md) · **简体中文**

<p align="center">
  <img src="assets/logo.svg" width="112" alt="cvorat 项目标识">
</p>

<h1 align="center">cvorat</h1>

<p align="center"><strong>字节级语言模型架构实验</strong></p>

<p align="center">
每个 notebook 只研究一个架构想法：窗口化与分组注意力、动态卷积、累积状态层、LoRA 分解和
RoPE 变体，全部建立在轻量的 PyTorch 层组件之上。
</p>

---

cvorat 是面向小型字节级语言模型的实验记录。它不把每个想法做成框架里的开关，而是让每个实验
保持为独立 notebook，共享两个小型库。目标是快速比较架构选择，而不是交付训练框架。

**状态：**公开的实验归档。notebook 本身就是记录；没有持续维护的版本、没有保存下来的基准表，
也没有锁定环境。归档说明见 [docs/project-status.md](docs/project-status.md)。

## 快速开始

完整训练需要 CUDA GPU 和 notebook 用到的依赖。

```bash
git clone https://github.com/ZisIsNotZis/cvorat.git
cd cvorat
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch numpy matplotlib triton flash-attn liger-kernel ipython jupyter
```

冒烟测试共享层组件：

```bash
python -c "import torch; from nn2 import Swi; print(Swi(64, 8, 8)(torch.randn(2, 8, 64)).shape)"
# torch.Size([2, 8, 8])
```

然后从仓库根目录打开 notebook，例如 [`12.ipynb`](12.ipynb)，从上到下运行。实验需要原始字节
语料：每个 token 一个 `uint8`。语料路径（例如 `../../py65536.bin`）硬编码在 notebook 里，
请改成自己的文件，或直接构造 loader：

```python
from nn2 import Dl

dl = Dl(a="/path/to/py.bin")   # 原始字节的 memmap
xb, yb = next(iter(dl))
```

并非每个 notebook 都需要全部依赖：`flash-attn` 用于注意力层和大多数编号 notebook，
`liger-kernel` 出现在后面的 Liger 变体中。

## 现在能做什么

- **字节级数据通路。**`nn2.Dl` 对 `uint8` 语料做 memmap，返回定长窗口；随机字节破坏施加在
  输入上，原始字节作为下一字节预测目标。
- **窗口化与分组注意力。**`nn.Attn` 及后续变体封装 FlashAttention，支持因果窗口、分组
  query/key/value 投影和可选的 ALiBi 斜率。
- **卷积与注意力并列。**`nn2.Cnv` 是窗口化、分组的卷积块；`cpa*` 实验把它和注意力、多项式层
  放进同一个残差块，有时还叠在池化层级下。`nn2.Dycnv` 是组件里的动态权重变体：为每个位置
  生成自己的卷积权重。
- **累积状态层。**`nn2.Cumavg` 和 `nn2.Cummax` 提供因果运行摘要；组件里还有一个带符号的
  累积衰减原型，配一个在对数空间计算衰减的 Triton 融合算子（`cumdecay_fuseker`）。
- **紧凑组件。**LoRA 分解线性层、SwiGLU、RoPE 和自定义 remap 函数，让组装变体不需要大型
  框架。
- **可读的实验记录。**68 个 notebook 覆盖模型形状、U-Net 式池化、Liger 融合、kernel 原型和
  少量支线实验。它们用来阅读和重跑，不作为包导入。

## 架构

两个共享层，每个实验一个 notebook：

```text
nn.py                  第一代层组件：注意力、SwiGLU、RoPE、remap、绘图
nn2.py                 第二代层组件：动态卷积、累积状态层、LoRA、
                       Triton 融合算子、字节级数据加载器
0.*–12*.ipynb          编号实验记录；每个 notebook 自包含
cpa*.ipynb             卷积 / 多项式 / 注意力的组合块
scratch-*.ipynb        支线实验：Liger kernel、推理服务、熵计算、累积 RoPE
docs/project-status.md 归档说明与边界
assets/logo.svg        项目标识
```

数据流刻意保持很小：

```text
原始字节 → Dl 窗口 + 随机字节破坏 → 用 nn / nn2 组件搭建模型
        → 下一字节交叉熵 → Plot / notebook 输出
```

Notebook 会刻意复制层定义：每一份都是搜索到该阶段时的模型快照。`nn.py` 和 `nn2.py` 是两个
导入面：大多数 notebook 从其中一个取辅助函数（`from nn import …`、`from nn2 import *`），同时
重写自己要研究的模型类。

## 如何开发

这个仓库是 notebook 归档，不是可安装的包。

- 用 Jupyter 从仓库根目录运行，这样 `import nn` / `import nn2` 才能解析到共享库。
  `nn2.py` 在模块加载时导入 `triton` 和 `IPython.display`，即使不碰这两者的块也需要它们已
  安装。
- `nn.py` 在模块加载时导入 FlashAttention；`nn2.py` 采用可选导入，只有 FlashAttention 注意力
  层需要它。
- 没有测试套件、CI、lockfile 或构建步骤。上面的冒烟测试是检查共享库能否导入并运行的最快
  方式。
- notebook 里的宽度、窗口大小、词表大小和语料路径都是硬编码的。要跑自己的实验，就改当前
  notebook 里的值；不要期待全局配置文件。
- 复用组件是预期的工作流。先把需要的块复制进新 notebook，等它稳定后，再只把有用的部分合并
  回 `nn2.py`。
- `nn2.py` 里有几个块已经和真正跑过的 notebook 副本脱节：`Cumdecay` 和 `cumdecay_fuse`
  引用了已不存在的名字（`cum`、`hr`、`normr`、`tl.prograc_id`）。两者不一致时以 notebook
  为准。

## 如何贡献

这是归档项目，功能请求不在范围内。欢迎复现报告和针对性修复：

1. 新开 issue，写清 notebook 名称、失败环境和完整 traceback。
2. 代码改动保持小，并指向 `nn.py`、`nn2.py` 或它修复的 notebook。
3. 不要大规模格式化或重写 notebook；其中重复的代码是记录的一部分。
4. 贡献按仓库的 AGPL-3.0 许可证接受。

## 路线图

### 已实现

- [x] 第一代层组件（`nn.py`）：窗口化/分组注意力、SwiGLU、RoPE 和 remap。
- [x] 第二代层组件（`nn2.py`）：动态卷积、LoRA 和累积状态层。
- [x] 带符号累积衰减原型，含 Triton 融合算子（在对数空间计算衰减；编号 notebook 未使用）。
- [x] 字节级 memmap 数据加载器与随机字节破坏目标。
- [x] 68 个 notebook，覆盖模型形状、池化层级、Liger 变体和 kernel 原型。
- [x] AGPL-3.0 许可证、项目标识和面向公众的文档。

### 计划中

- [ ] 整理实验结果。此前没有记录结果，所以这里不作任何结论。
- [ ] 让一次运行可以端到端复现：锁定环境、固定随机种子、记录语料来源。
- [ ] 加入一个小型示例语料，让冒烟路径不依赖私有文件。
- [ ] 把已经稳定的重复层定义合并进 `nn2.py`。
- [ ] 把最有用的模型族各整理成一个可运行配方。

### 未来设想

长期价值不在于再写一个模型实现，而在于用同一份语料和同一套评测，对小架构想法做可复现的
比较。notebook 可以继续作为实验台，共享库成为稳定接口，结果整理成一份简短、诚实的报告。
在报告出现之前，这个仓库是证据，不是结论。

## 许可证

[GNU Affero 通用公共许可证 v3.0](LICENSE)（AGPL-3.0）。
