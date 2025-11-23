#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查币安API设置是否有效
"""

import streamlit as st
import requests

def check_binance_api():
    """检查币安API密钥是否已配置且能够连接"""
    try:
        # 检查是否有API密钥配置
        if "binance_api_key" in st.secrets:
            api_key = st.secrets["binance_api_key"]
            # 简单测试API连接
            test_url = "https://api.binance.com/api/v3/ping"
            headers = {"X-MBX-APIKEY": api_key}
            response = requests.get(test_url, headers=headers, timeout=5)
            if response.status_code == 200:
                return True, "API连接成功"
            else:
                return False, f"API连接测试失败: {response.status_code} {response.reason}"
        else:
            return False, "未配置币安API密钥"
    except Exception as e:
        return False, f"API测试出错: {str(e)}"

if __name__ == "__main__":
    success, message = check_binance_api()
    print(f"API状态: {'成功' if success else '失败'}")
    print(f"消息: {message}")