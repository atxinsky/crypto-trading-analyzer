#!/bin/bash

echo "正在安装币安交易数据高级分析系统..."

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.7或更高版本"
    exit 1
fi

# 创建虚拟环境
echo "创建虚拟环境..."
python3 -m venv venv

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "安装依赖包..."
pip install -r requirements.txt

echo "安装完成！"
echo "运行以下命令启动应用："
echo "source venv/bin/activate"
echo "streamlit run upgraded.py"