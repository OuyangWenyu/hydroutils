# hydroutils

[![image](https://img.shields.io/pypi/v/hydroutils.svg)](https://pypi.python.org/pypi/hydroutils)
[![image](https://img.shields.io/conda/vn/conda-forge/hydroutils.svg)](https://anaconda.org/conda-forge/hydroutils)
[![image](https://pyup.io/repos/github/OuyangWenyu/hydroutils/shield.svg)](https://pyup.io/repos/github/OuyangWenyu/hydroutils)
[![Python Version](https://img.shields.io/pypi/pyversions/hydroutils.svg)](https://pypi.org/project/hydroutils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**用于水文建模和分析的实用工具函数集合**

Hydroutils 是一个专为水文建模工作流设计的现代 Python 包，提供统计分析、数据可视化、文件处理、时间操作和单位转换功能，专门针对水文研究和应用进行优化。

**本项目仍在开发中，API 可能会发生变化。**

- **免费软件**: MIT 许可证
- **文档**: https://OuyangWenyu.github.io/hydroutils
- **源代码**: https://github.com/OuyangWenyu/hydroutils
- **PyPI 包**: https://pypi.org/project/hydroutils/

## ✨ 功能特性

### 📊 统计分析 (`hydro_stat`)
- **动态指标函数**: 自动生成的统计函数（NSE、RMSE、MAE 等）
- **多维分析**: 支持 2D/3D 数组进行流域尺度分析
- **HydroErr 集成**: 通过 HydroErr 包提供标准化水文指标
- **NaN 处理**: 灵活的缺失数据处理策略（'no'、'sum'、'mean'）
- **运行时指标添加**: 使用 `add_metric()` 动态添加自定义指标

### 📈 可视化 (`hydro_plot`)
- **地理空间绘图**: Cartopy 集成支持基于地图的可视化
- **中文字体支持**: 自动配置中文文本渲染字体
- **统计图表**: ECDF、箱线图、热力图、相关矩阵
- **水文专业图表**: 流量历时曲线、单位线、降水图
- **可定制样式**: 丰富的颜色、样式和格式配置选项

### 📁 文件操作 (`hydro_file`)
- **JSON 序列化**: 使用 `NumpyArrayEncoder` 支持 NumPy 数组
- **云存储**: S3 和 MinIO 集成用于远程数据访问
- **ZIP 处理**: 嵌套 ZIP 文件提取和管理
- **缓存管理**: 自动缓存目录创建和管理
- **异步操作**: 异步数据检索功能

### ⏰ 时间操作 (`hydro_time`)
- **UTC 计算**: 根据坐标计算时区偏移
- **日期解析**: 灵活的日期字符串解析和处理
- **时间范围操作**: 交集、生成和验证
- **间隔检测**: 自动时间间隔识别

### 🏷️ 单位转换 (`hydro_units`)
- **流量单位**: 水文变量的综合单位转换
- **时间间隔检测**: 自动检测和验证时间间隔
- **单位兼容性**: 单位一致性验证函数
- **Pint 集成**: 使用 pint 和 pint-xarray 处理物理单位

### 🌊 事件分析 (`hydro_event`)
- **水文事件检测**: 洪峰流量、枯水流量和洪水事件识别
- **事件特征化**: 持续时间、规模和时机分析
- **多变量事件**: 多个变量的复合事件分析

### ☁️ 云集成 (`hydro_s3`)
- **AWS S3 支持**: 直接集成 Amazon S3 服务
- **MinIO 兼容**: 本地和私有云存储解决方案
- **异步下载**: 高性能异步数据检索
- **凭证管理**: 安全的凭证处理和配置

### 📝 日志记录 (`hydro_log`)
- **富文本控制台输出**: 彩色和格式化控制台日志
- **进度跟踪**: 高级进度条和状态指示器
- **调试支持**: 全面的调试和错误报告

## 🚀 快速开始

### 安装

```bash
# 从 PyPI 安装
pip install hydroutils

# 使用 uv 安装开发依赖（推荐）
pip install uv
uv add hydroutils

# 开发环境设置
git clone https://github.com/OuyangWenyu/hydroutils.git
cd hydroutils
uv sync --all-extras --dev
```

### 基本用法

```python
import hydroutils
import numpy as np

# 统计分析
obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
sim = np.array([1.1, 2.1, 2.9, 3.9, 5.1])

# 计算 Nash-Sutcliffe 效率系数
nse_value = hydroutils.nse(obs, sim)
print(f"NSE: {nse_value:.3f}")

# 一次计算多个指标
metrics = hydroutils.stat_error(obs, sim)
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"MAE: {metrics['mae']:.3f}")

# 可视化
import matplotlib.pyplot as plt
fig, ax = hydroutils.plot_ecdf([obs, sim], 
                               labels=['观测值', '模拟值'],
                               colors=['blue', 'red'])
plt.show()

# 单位转换
flow_cms = 100.0  # 立方米/秒
flow_cfs = hydroutils.streamflow_unit_conv(flow_cms, 'cms', 'cfs')
print(f"流量: {flow_cms} m³/s = {flow_cfs:.2f} ft³/s")

# 时间操作
from datetime import datetime
utc_offset = hydroutils.get_utc_offset_from_coordinates(39.9, 116.4)  # 北京
print(f"北京 UTC 偏移: {utc_offset} 小时")
```

## 🛠️ 开发

### 设置开发环境

```bash
# 克隆仓库
git clone https://github.com/OuyangWenyu/hydroutils.git
cd hydroutils

# 安装 UV（现代 Python 包管理器）
pip install uv

# 设置开发环境
uv sync --all-extras --dev
uv run pre-commit install

# 或者使用 Makefile
make setup-dev
```

### 开发命令

```bash
# 运行测试
uv run pytest                    # 基本测试运行
uv run pytest --cov=hydroutils   # 带覆盖率
make test-cov                    # 带 HTML 覆盖率报告

# 代码格式化和检查
uv run black .                   # 格式化代码
uv run ruff check .              # 检查代码
uv run ruff check --fix .        # 修复检查问题
make format                      # 格式化和检查一起

# 类型检查
uv run mypy hydroutils
make type-check

# 文档
uv run mkdocs serve              # 本地服务文档
make docs-serve

# 构建和发布
uv run python -m build           # 构建包
make build
```

### 项目结构

```
hydroutils/
├── hydroutils/
│   ├── __init__.py              # 包初始化和导出
│   ├── hydro_event.py           # 水文事件分析
│   ├── hydro_file.py            # 文件 I/O 和云存储
│   ├── hydro_log.py             # 日志记录和控制台输出
│   ├── hydro_plot.py            # 可视化函数
│   ├── hydro_s3.py              # AWS S3 和 MinIO 集成
│   ├── hydro_stat.py            # 统计分析引擎
│   ├── hydro_time.py            # 时间序列工具
│   └── hydro_units.py           # 单位转换和验证
├── tests/                       # 综合测试套件
├── docs/                        # MkDocs 文档
├── pyproject.toml               # 现代 Python 项目配置
├── Makefile                     # 开发便利命令
└── uv.lock                      # UV 包管理器锁定文件
```

## 🤝 贡献

我们欢迎贡献！请查看我们的[贡献指南](docs/contributing.md)了解详情。

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 进行更改
4. 运行测试和检查 (`make check-all`)
5. 提交更改 (`git commit -m 'Add amazing feature'`)
6. 推送到分支 (`git push origin feature/amazing-feature`)
7. 打开 Pull Request

## 📖 文档

完整文档可在 [https://OuyangWenyu.github.io/hydroutils](https://OuyangWenyu.github.io/hydroutils) 获取，包括：

- **API 参考**: 完整的函数和类文档
- **用户指南**: 逐步教程和示例
- **贡献指南**: 开发设置和贡献指南
- **FAQ**: 常见问题和故障排除

## 🏗️ 要求

- **Python**: >=3.10
- **核心依赖**: numpy, pandas, matplotlib, seaborn
- **科学计算**: scipy, HydroErr, numba
- **可视化**: cartopy（用于地理空间图表）
- **云存储**: boto3, minio, s3fs
- **工具**: tqdm, rich, xarray, pint

## 📄 许可证

本项目根据 MIT 许可证授权 - 有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **HydroErr**: 提供标准化水文误差指标
- **Cookiecutter**: 项目模板来自 [giswqs/pypackage](https://github.com/giswqs/pypackage)
- **科学 Python 生态系统**: NumPy, SciPy, Matplotlib, Pandas

## 📞 支持

- **问题**: [GitHub Issues](https://github.com/OuyangWenyu/hydroutils/issues)
- **讨论**: [GitHub Discussions](https://github.com/OuyangWenyu/hydroutils/discussions)
- **邮箱**: wenyuouyang@outlook.com

---

**为水文建模社区打造 ❤️**