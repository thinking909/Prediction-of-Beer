# 基于化合物数据的啤酒风味预测与风格生成

## 1. 目录结构

```text
项目文档/                （本目录）
├─ 附件/                 原始数据 1.csv–8.xlsx + 清洗/还原/映射代码
├─ Q1/                   问题一：数据校验、预测、稳定性、交付、图表
├─ Q2/                   问题二：共变网络、重要性、共识、drop-column、风格、图表
├─ Q3/                   问题三：生成、约束、感官画像、迁移、评价、图表
├─ wheels/               Windows 64-bit + Python 3.11 的离线 wheel 包
├─ run_all.py            一键复现入口
├─ requirements.txt      Python 依赖
├─ environment.yml       conda 环境定义
├─ install_env.bat       离线安装环境（创建 .venv）
├─ run.bat               双击运行一键复现
└─ README.md             本文件
```

`解答结果/` 和 `论文/` 位于本目录的上一级（项目根目录）；`论文/` 下包含 `main.tex` 和配套的 `AI工具使用情况说明.tex`。

运行后会新增：

```text
附件/清洗后/             清洗后的化学、感官、评论数据及审计表
Q1/results/  Q1/figures/
Q2/results/  Q2/figures/
Q3/results/  Q3/figures/
reproduce.log            最近一次运行日志（可删除）
```

项目根目录下的 `解答结果/` 是本次完整运行的结果快照：`第一问结果.xlsx`、`第二问结果.xlsx`、`第三问结果.xlsx`，以及 `第三问_结果表/` 的 CSV 版本；重新运行 `run_all.py` 会在 `Q1/Q2/Q3` 下重新生成结果，但不会自动覆盖该快照目录。

## 2. 环境

推荐使用 Python 3.11+。实测环境：

```text
Windows 10
Python 3.11.4 (Anaconda)
numpy 1.24.3, pandas 1.5.3, scipy 1.10.1
scikit-learn 1.3.0, matplotlib 3.7.1
openpyxl 3.0.10, networkx 3.1, joblib 1.2.0
```

安装（推荐，离线）：

```text
双击 install_env.bat
```

它会创建项目内 `.venv`，并从 `wheels/` 离线安装全部 wheel，
不会联网下载。之后双击 `run.bat` 即可。

如果已装 Anaconda，也可以用：

```text
conda env create -f environment.yml
conda activate beer-flavor
python run_all.py
```

如果换到 macOS/Linux 或别的 Python 版本，`wheels/` 不适用，请联网执行：

```text
pip install -r requirements.txt
```

详见 `环境说明.txt`。不需要 xgboost；当前代码不使用它。

## 3. Windows 双击一键复现
1. 安装 **Python 3.11 64-bit**（或 Anaconda）。
2. 双击 **`install_env.bat`**：创建项目内 `.venv`，从 `wheels/` 离线安装依赖，不联网。
3. 双击 **`run.bat`**：自动调用 `.venv\Scripts\python.exe run_all.py`，开始完整复现。

说明：

- `install_env.bat` 只需第一次运行；以后直接双击 `run.bat`。
- `run.bat` 支持透传参数，例如 `run.bat --stages q3`。
- 如果窗口一闪而过，右键“在终端中打开”，执行 `cmd /k install_env.bat` 或 `cmd /k run.bat` 查看具体报错。
- `.venv` 是本机生成的环境，体积较大；复制到其他电脑时建议删除 `.venv` 后重新运行 `install_env.bat`，仍然不需要联网下载。
- 如果目标电脑没有 Python 3.11，`.bat` 无法创建环境，需要先安装 Python 或改用 Anaconda。

## 4. 命令行一键复现

在本目录（`项目文档`）执行：

```text
python run_all.py
```

约 2.5–3.5 小时，主要耗时是 Q1 的稳定性协议。

常用选项：

```text
python run_all.py --list
python run_all.py --stages clean,q1
python run_all.py --stages q3
python run_all.py --skip clean
python run_all.py --no-hybrid
python run_all.py --with-ablations
```

阶段说明：

| 阶段 | 内容 |
|---|---|
| `clean` | 运行 `附件` 清洗、还原、预处理补充、评论映射、异常值图 |
| `q1` | 数据校验、主协议、稳定性、尺度消融、线性裁剪、交付、跨源、Q1 图表 |
| `q2` | 共变网络、单变量、稳定性、重要性、共识、drop-column、风格、最终 Top-3、Q2 图表 |
| `q3` | v6 生成、缺陷约束、感官画像、迁移、评价、C2ST null、spiking、Q3 图表 |

Q3 默认还会运行 Hybrid 生成器并恢复 v6 交付状态；`--no-hybrid` 可跳过。`--with-ablations` 会额外运行 Q3 生成器消融，耗时明显更长。

## 5. 输出

- 清洗数据：`附件/清洗后/`
- 问题一：`Q1/results/`、`Q1/figures/`
- 问题二：`Q2/results/`、`Q2/figures/`
- 问题三：`Q3/results/`、`Q3/figures/`
- 运行日志：`reproduce.log`

脚本不生成 `.md` 文件，不编译 LaTeX，也不依赖任何外部论文/代码。

## 6. 注意事项

- 清洗和各问脚本均使用项目相对路径，整个目录可直接复制到其他电脑运行。
- Q3 `step1_prepare.py` 使用 `QuantileTransformer(subsample=len(Xtr))`，兼容 scikit-learn 1.3 与 1.8。
- 原始数据只读；所有清洗结果、模型结果和图表均为运行时生成，可随时删除后重跑。
- 若 `python` 指向 Windows 应用商店占位程序，请改用 Anaconda 的 `python.exe`，或先激活对应环境。