@echo off
echo 正在安装币安交易数据高级分析系统...

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.7或更高版本
    pause
    exit /b 1
)

REM 创建虚拟环境
echo 创建虚拟环境...
python -m venv venv

REM 激活虚拟环境
echo 激活虚拟环境...
call venv\Scripts\activate.bat

REM 安装依赖
echo 安装依赖包...
pip install -r requirements.txt

echo 安装完成！
echo 运行以下命令启动应用：
echo venv\Scripts\activate.bat
echo streamlit run upgraded.py
pause