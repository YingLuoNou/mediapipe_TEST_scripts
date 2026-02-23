## 🛠 开发环境准备
本项目使用 uv 进行项目管理和依赖控制。uv 是一个极其快速的 Python 包管理工具，旨在替代 pip、poetry 和 venv。

### 1. 安装 uv
如果你还没有安装 uv，可以通过以下命令安装：

macOS / Linux:

```Bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Windows (PowerShell):

```PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
### 2. 初始化环境
克隆仓库后，在项目根目录下运行以下命令，uv 会自动创建虚拟环境并同步所有依赖：

```Bash
uv sync
```
## 🚀 常用操作
### 运行程序
使用 uv run 可以在自动激活的虚拟环境中执行脚本：

```Bash
uv run main.py
```
### 添加新依赖
```Bash
# 添加普通依赖
uv add requests

# 添加开发环境依赖
uv add --dev pytest
```
### 更新依赖
```Bash
uv lock --upgrade
```
