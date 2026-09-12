# Overleaf 编译说明

1. 将本目录压缩包上传到 Overleaf，或解压后上传全部文件。
2. 在 Overleaf 的 `Menu -> Compiler` 中选择 `XeLaTeX`。
3. 主文档设置为 `main.tex`，点击 `Recompile`。
4. 工程使用 `ctexart` 的 Fandol 字体配置，不依赖本地宋体或黑体。

目录说明：

- `main.tex`：16 页论文初稿；
- `figures/`：正文引用的 7 幅 PNG 图片；
- `code/`：完整求解、结果分析和模型检验程序；
- `data/`：处理后特征库、逐企业策略及检验结果；
- `main.pdf`：本地 XeLaTeX 实际编译得到的预览稿。

本地复现：

```bash
xelatex main.tex
xelatex main.tex
```

连续编译两次用于解析交叉引用和文献编号。
