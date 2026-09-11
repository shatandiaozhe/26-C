# 2020 C题模型求解代码运行说明

## 1. 文件位置

主程序为 `model_solver.py`。运行前应保留项目原有的 `C` 文件夹以及上一阶段生成的：

`outputs/data_stage/2020C_数据预处理与特征库.xlsx`

## 2. 安装Python

推荐Python 3.10或以上版本。在项目文件夹打开PowerShell或命令提示符。

## 3. 安装依赖

```bash
python -m pip install numpy pandas openpyxl
```

本程序不要求安装SciPy、Scikit-learn、Matplotlib或商业优化软件。

## 4. 运行

```bash
python code/model_solver.py
```

运行完成后查看：

`outputs/model_solution/`

其中包含三问的CSV结果、模型评价JSON、Markdown摘要以及6张SVG图。

## 5. 参数调整

所有参数位于主程序开头的 `PARAM` 字典。

- `lgd`：必须在0到1之间；
- `cvar_alpha`：必须严格在0到1之间；
- `uncertainty_eta`：越大，对无信贷记录企业越保守；
- `risk_aversion`：越大，越重视风险控制；
- `mc_scenarios`：越大，CVaR越稳定，但运行时间越长；
- `question1_budget_wan`：问题1题目未给出总额，可修改后重新运行。

## 6. 常见错误

1. `FileNotFoundError`：检查特征库是否位于规定位置。
2. `ModuleNotFoundError`：重新执行依赖安装命令。
3. Excel文件被占用：关闭正在打开的相关工作簿后重新运行。
4. 结果与本次略有差异：检查是否修改了随机种子或输入数据。

