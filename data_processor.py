import pandas as pd
import os
from datetime import datetime
import numpy as np

def load_and_merge_data(file_paths):
    """
    加载多个CSV文件并合并为一个DataFrame
    """
    all_data = []
    
    for file_path in file_paths:
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 处理币安导出的CSV文件格式
            # 确保Time(UTC)列被正确解析为日期时间格式
            if 'Time(UTC)' in df.columns:
                df['Time(UTC)'] = pd.to_datetime(df['Time(UTC)'], format='%Y/%m/%d %H:%M', errors='coerce')
                
                # 创建标准化列名的映射
                df['timestamp'] = df['Time(UTC)']  # 保持原始列的同时添加映射列
                df['qty'] = df['Quantity']
                df['realized_pnl'] = df['Realized Profit']
                
                # 从Fee列中提取金额和币种
                if 'Fee' in df.columns:
                    # 提取Fee金额
                    df['fee'] = df['Fee'].str.extract(r'(-?\d+\.?\d*)').astype(float)
                    
                    # 提取Fee币种
                    df['fee_currency'] = df['Fee'].str.extract(r'[0-9\.\-]+ ([A-Za-z]+)')
                    
                    # 处理不包含币种的情况
                    df['fee_currency'] = df['fee_currency'].fillna('UNKNOWN')
            
            # 添加原始文件名作为标识（可选）
            df['Source_File'] = os.path.basename(file_path)
            
            all_data.append(df)
            print(f"成功加载文件: {file_path}, 记录数: {len(df)}")
        except Exception as e:
            print(f"加载文件失败 {file_path}: {str(e)}")
    
    if not all_data:
        return None
    
    # 合并所有数据
    merged_data = pd.concat(all_data, ignore_index=True)
    
    # 按时间排序
    merged_data.sort_values('timestamp', inplace=True)
    
    return merged_data

def clean_and_process_data(df):
    """
    清洗和处理数据
    """
    if df is None or df.empty:
        return None
    
    # 创建交易日期列（不含时间）
    df['Trade_Date'] = df['timestamp'].dt.date
    
    # 确保Amount列是数值型
    if 'Amount' in df.columns and pd.api.types.is_object_dtype(df['Amount']):
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    # 确保Quantity/qty列是数值型
    if 'Quantity' in df.columns and pd.api.types.is_object_dtype(df['Quantity']):
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['qty'] = df['Quantity']  # 确保qty列存在
    
    # 确保Price列是数值型
    if 'Price' in df.columns and pd.api.types.is_object_dtype(df['Price']):
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # 确保Realized Profit/realized_pnl列是数值型
    if 'Realized Profit' in df.columns:
        if pd.api.types.is_object_dtype(df['Realized Profit']):
            df['Realized Profit'] = pd.to_numeric(df['Realized Profit'], errors='coerce')
        df['realized_pnl'] = df['Realized Profit']  # 确保realized_pnl列存在
    
    # 确保fee列存在
    if 'Fee' in df.columns and 'fee' not in df.columns:
        df['fee'] = df['Fee'].str.extract(r'(-?\d+\.?\d*)').astype(float)
    
    return df

def aggregate_daily_trades(df):
    """
    按币种和日期聚合交易数据，分别统计买入和卖出
    """
    if df is None or df.empty:
        return None
    
    # 按币种、日期和交易方向分组聚合
    grouped = df.groupby(['Symbol', 'Trade_Date', 'Side']).agg({
        'qty': 'sum',
        'Amount': 'sum',
        'fee': 'sum',
        'realized_pnl': 'sum',
        'Uid': 'count'  # 计算交易次数
    }).reset_index()
    
    # 重命名计数列
    grouped.rename(columns={'Uid': 'Trade_Count'}, inplace=True)
    
    # 计算平均价格
    grouped['Average_Price'] = grouped['Amount'] / grouped['qty']
    
    return grouped

def calculate_pnl_metrics(df):
    """
    计算盈亏指标
    """
    if df is None or df.empty:
        return None
    
    # 计算每个交易的盈亏
    df['PnL'] = df['realized_pnl'] - df['fee']
    
    # 计算胜率
    total_trades = len(df)
    winning_trades = len(df[df['PnL'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 计算盈亏比
    avg_win = df[df['PnL'] > 0]['PnL'].mean() if winning_trades > 0 else 0
    losing_trades = len(df[df['PnL'] < 0])
    avg_loss = abs(df[df['PnL'] < 0]['PnL'].mean()) if losing_trades > 0 else 0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # 计算总体盈亏
    total_pnl = df['PnL'].sum()
    
    # 获取最盈利和最亏损的交易
    df_sorted = df.sort_values('PnL', ascending=False)
    top_profitable = df_sorted.head(5)
    top_unprofitable = df_sorted.tail(5)
    
    # 返回指标
    metrics = {
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'top_profitable': top_profitable,
        'top_unprofitable': top_unprofitable.iloc[::-1]  # 反转顺序以便显示最亏损的在前
    }
    
    return metrics

def save_processed_data(df, output_path):
    """
    保存处理后的数据到指定路径
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"数据已保存至: {output_path}")
        return True
    except Exception as e:
        print(f"保存数据失败: {str(e)}")
        return False

def get_OHLC_data(df, symbol=None, start_date=None, end_date=None):
    """
    从交易数据中生成OHLC数据用于K线图
    """
    if df is None or df.empty:
        return None
    
    # 过滤数据
    filtered_df = df.copy()
    if symbol:
        filtered_df = filtered_df[filtered_df['Symbol'] == symbol]
    
    if start_date:
        filtered_df = filtered_df[filtered_df['timestamp'] >= pd.to_datetime(start_date)]
    
    if end_date:
        filtered_df = filtered_df[filtered_df['timestamp'] <= pd.to_datetime(end_date)]
    
    if filtered_df.empty:
        return None
    
    # 按日期重采样获取OHLC数据
    filtered_df.set_index('timestamp', inplace=True)
    ohlc = filtered_df['Price'].resample('D').ohlc()
    
    # 添加成交量
    volume = filtered_df['qty'].resample('D').sum()
    ohlc['volume'] = volume
    
    return ohlc.reset_index()

def prepare_trade_markers(df, symbol=None, start_date=None, end_date=None):
    """
    准备买入/卖出标记用于K线图
    """
    if df is None or df.empty:
        return None, None
    
    # 过滤数据
    filtered_df = df.copy()
    if symbol:
        filtered_df = filtered_df[filtered_df['Symbol'] == symbol]
    
    if start_date:
        filtered_df = filtered_df[filtered_df['timestamp'] >= pd.to_datetime(start_date)]
    
    if end_date:
        filtered_df = filtered_df[filtered_df['timestamp'] <= pd.to_datetime(end_date)]
    
    if filtered_df.empty:
        return None, None
    
    # 分离买入和卖出交易
    buy_trades = filtered_df[filtered_df['Side'] == 'BUY']
    sell_trades = filtered_df[filtered_df['Side'] == 'SELL']
    
    return buy_trades, sell_trades
