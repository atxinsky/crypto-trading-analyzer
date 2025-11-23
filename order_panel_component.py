# order_panel_component.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from decimal import Decimal, InvalidOperation
import math # Import math module

def safe_to_decimal(x):
    """Safely convert input to Decimal, handling potential errors."""
    if pd.isna(x):
        return Decimal(0)
    try:
        # Convert various types to string first for consistent Decimal conversion
        return Decimal(str(x))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(0) # Return 0 if conversion fails

def create_order_panel(filtered_trades, symbol=None):
    """
    创建订单面板 (假设输入数据已标准化)

    参数:
    filtered_trades (DataFrame): 经过筛选和标准化的交易记录
    symbol (str): 当前选择的交易对 (用于筛选)
    """
    if filtered_trades.empty:
        st.info("没有符合条件的交易记录")
        return

    # --- 使用标准化的列名 ---
    time_col = 'timestamp'
    symbol_col = 'symbol'
    side_col = 'side'
    price_col = 'price'
    qty_col = 'qty'         # 使用处理后的数值数量列
    pnl_col = 'pnl'         # 使用处理后的统一PnL列
    amount_col = 'amount'   # 使用处理后的数值金额列 (可能是Quote或Base，取决于数据类型)
    base_asset_col = 'base_asset'
    quote_asset_col = 'quote_asset'
    fee_col = 'fee'         # 使用处理后的数值费用列

    # 检查必需的列是否存在 (pnl_col is checked later as it might be calculated)
    required_cols_check = [time_col, symbol_col, side_col, price_col, qty_col, amount_col, base_asset_col, quote_asset_col, fee_col]
    if not all(col in filtered_trades.columns for col in required_cols_check):
        missing = [col for col in required_cols_check if col not in filtered_trades.columns]
        st.error(f"输入数据缺少必需的标准化列。需要: {required_cols_check}, 实际缺少: {missing}")
        return

    # --- 如果指定了symbol，则进一步过滤 ---
    if symbol:
        symbol_trades = filtered_trades[filtered_trades[symbol_col] == symbol].copy()
        if symbol_trades.empty:
             st.warning(f"未找到交易对 '{symbol}' 的交易记录。")
             return
    else:
        symbol_trades = filtered_trades.copy() # Use all data if no specific symbol

    # Ensure pnl_col exists and is Decimal
    if pnl_col not in symbol_trades.columns:
        st.error(f"必需的 PnL 列 '{pnl_col}' 在过滤后的数据中不存在。")
        return
    symbol_trades[pnl_col] = symbol_trades[pnl_col].apply(safe_to_decimal)
    symbol_trades[price_col] = symbol_trades[price_col].apply(safe_to_decimal)
    symbol_trades[qty_col] = symbol_trades[qty_col].apply(safe_to_decimal)
    symbol_trades[amount_col] = symbol_trades[amount_col].apply(safe_to_decimal)
    symbol_trades[fee_col] = symbol_trades[fee_col].apply(safe_to_decimal)


    # --- 交易对概览 ---
    st.write("#### 交易对概览")
    try:
        # Group by symbol and side
        grouped = symbol_trades.groupby([symbol_col, side_col]).agg(
            总数量=(qty_col, 'sum'),
            总盈亏=(pnl_col, 'sum'),
            交易次数=(time_col, 'count') # Count any non-null column
        ).reset_index()

        if not grouped.empty:
            # Pivot for better display
            summary = grouped.pivot_table(
                index=symbol_col,
                columns=side_col,
                values=['总数量', '总盈亏', '交易次数'],
                aggfunc='sum' # Use sum here, as agg already did the sum within group
            ).reset_index()
            summary.columns = [f"{col[0]}_{col[1]}" if col[1]!='' else col[0] for col in summary.columns]
            summary.fillna(0, inplace=True) # Replace NaN with 0 after pivot

            # Add Net PnL (Ensure columns exist before calculation)
            buy_pnl_col = f'总盈亏_BUY'
            sell_pnl_col = f'总盈亏_SELL'
            if sell_pnl_col in summary.columns and buy_pnl_col in summary.columns:
                 summary['净盈亏'] = summary[sell_pnl_col] + summary[buy_pnl_col] # BUY PnL is already negative for spot cost
            elif sell_pnl_col in summary.columns:
                 summary['净盈亏'] = summary[sell_pnl_col]
            elif buy_pnl_col in summary.columns:
                 summary['净盈亏'] = summary[buy_pnl_col] # Should be negative for spot
            else:
                 summary['净盈亏'] = Decimal(0)


            # Add Asset Info (Example using first row of the group)
            # Ensure the columns exist before trying to group and access them
            if base_asset_col in symbol_trades.columns and quote_asset_col in symbol_trades.columns:
                asset_info = symbol_trades.groupby(symbol_col)[[base_asset_col, quote_asset_col]].first().reset_index()
                summary = pd.merge(summary, asset_info, on=symbol_col, how='left')
                base_asset_display_col = base_asset_col
                quote_asset_display_col = quote_asset_col
            else:
                base_asset_display_col = '基础资产' # Placeholder names
                quote_asset_display_col = '计价资产'
                summary[base_asset_display_col] = 'N/A'
                summary[quote_asset_display_col] = 'N/A'


            # Reorder and select columns for display
            display_cols_order = [symbol_col, base_asset_display_col, quote_asset_display_col, '净盈亏']
            optional_cols = [
                '总数量_BUY', '总数量_SELL', '总盈亏_BUY', '总盈亏_SELL',
                '交易次数_BUY', '交易次数_SELL'
            ]
            # Only include optional columns if they exist in the summary dataframe
            display_cols_order.extend([col for col in optional_cols if col in summary.columns])

            # Select only the columns that actually exist
            final_display_cols = [col for col in display_cols_order if col in summary.columns]

            st.dataframe(summary[final_display_cols], use_container_width=True)
        else:
            st.info("没有足够的交易数据进行概览。")

    except Exception as e:
        st.error(f"创建交易对概览时出错: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")


    # --- 详细交易记录 (只在选择了特定交易对时显示) ---
    if symbol:
        st.write(f"#### {symbol} 详细交易记录")
        try:
            # Select columns for the detailed table
            details_cols = [time_col, side_col, price_col, qty_col, pnl_col, amount_col, fee_col, 'fee_currency']
            details_cols_present = [col for col in details_cols if col in symbol_trades.columns]
            details = symbol_trades[details_cols_present].copy()

            # Format timestamp
            details[time_col] = pd.to_datetime(details[time_col]).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Combine Fee and Currency for display
            if 'fee' in details.columns and 'fee_currency' in details.columns:
                 details['费用'] = details.apply(lambda row: f"{row['fee']:.8f} {row['fee_currency']}" if pd.notna(row['fee_currency']) else f"{row['fee']:.8f}", axis=1)
            elif 'fee' in details.columns:
                 details['费用'] = details['fee'].apply(lambda x: f"{x:.8f}")


            # Rename columns for display
            col_mapping = {
                time_col: '时间',
                side_col: '方向',
                price_col: '价格',
                qty_col: '数量',
                pnl_col: '盈亏',
                amount_col: '金额' # This is quote value for spot, base value for futures
            }
            details = details.rename(columns=col_mapping)

            # Select final display columns including the new '费用'
            final_details_cols = ['时间', '方向', '价格', '数量', '金额', '费用', '盈亏']
            final_details_cols_present = [col for col in final_details_cols if col in details.columns]
            details_display = details[final_details_cols_present]


            # Sort by time descending
            details_display = details_display.sort_values('时间', ascending=False)

            # Apply styling
            def highlight_side(val):
                if val == 'BUY': return 'background-color: rgba(0, 255, 0, 0.2)'
                elif val == 'SELL': return 'background-color: rgba(255, 0, 0, 0.2)'
                return ''
            def highlight_pnl(val):
                # Check if it's already Decimal before formatting
                if isinstance(val, Decimal):
                     val_dec = val
                else:
                     # Try converting if not Decimal, default to 0 on error
                     try: val_dec = Decimal(str(val))
                     except: val_dec = Decimal(0)

                if val_dec > 0: return 'color: green; font-weight: bold;'
                elif val_dec < 0: return 'color: red; font-weight: bold;'
                return ''

            # Apply formatting after potential highlights
            styled_details = details_display.style.applymap(highlight_side, subset=['方向'] if '方向' in details_display else None)\
                                          .applymap(highlight_pnl, subset=['盈亏'] if '盈亏' in details_display else None)\
                                          .format({ # Apply number formatting
                                              '价格': "{:.8f}",
                                              '数量': "{:.8f}",
                                              '盈亏': "{:.8f}", # Format PnL as well
                                              '金额': "{:.8f}"
                                            }, na_rep='-')


            st.dataframe(styled_details, use_container_width=True, height=400) # Add height for scroll

             # --- PnL Visualization for the selected symbol ---
            if not details_display.empty and '盈亏' in details_display.columns and '时间' in details_display.columns:
                st.write("#### 盈亏分布")
                fig = go.Figure()

                # Ensure '盈亏' is numeric (Decimal) for cumsum
                details_display['盈亏_numeric'] = details_display['盈亏'].apply(safe_to_decimal)
                cumulative_pnl = details_display['盈亏_numeric'].cumsum()

                fig.add_trace(go.Scatter(x=details_display['时间'], y=cumulative_pnl, mode='lines+markers', name='累计盈亏', line=dict(color='royalblue', width=2)))

                colors = ['green' if pnl >= 0 else 'red' for pnl in details_display['盈亏_numeric']]
                fig.add_trace(go.Bar(x=details_display['时间'], y=details_display['盈亏_numeric'], marker_color=colors, name='交易盈亏'))

                fig.update_layout(title=f"{symbol} 盈亏分析", xaxis_title="交易时间", yaxis_title="盈亏", height=400,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)


        except Exception as e:
            st.error(f"创建 {symbol} 详细记录时出错: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")

def generate_trade_statistics(filtered_trades, symbol=None):
    """生成综合交易统计信息 (假设输入数据已标准化)"""
    if filtered_trades.empty: return None
    symbol_col = 'symbol'; side_col = 'side'; pnl_col = 'pnl'
    if symbol: df = filtered_trades[filtered_trades[symbol_col] == symbol].copy()
    else: df = filtered_trades.copy()
    if df.empty: return None
    if pnl_col not in df.columns: st.error(f"PnL column '{pnl_col}' not found for statistics."); return None
    df[pnl_col] = df[pnl_col].apply(safe_to_decimal) # Ensure PnL is Decimal
    try:
        total_trades = len(df)
        profitable_trades = df[df[pnl_col] > 0]; losing_trades = df[df[pnl_col] < 0]
        win_count = len(profitable_trades); loss_count = len(losing_trades)
        total_pnl = df[pnl_col].sum()
        max_profit = df[pnl_col].max() if not df.empty else Decimal(0)
        max_loss = df[pnl_col].min() if not df.empty else Decimal(0)

        # Use float for stats that involve division or std dev if Decimal causes issues
        pnl_float = df[pnl_col].astype(float)
        avg_pnl_float = pnl_float.mean() if total_trades > 0 else 0.0
        std_dev_pnl_float = pnl_float.std() if total_trades > 1 else 0.0

        win_rate = float(win_count) / float(total_trades) if total_trades > 0 else 0.0
        avg_profit_float = float(profitable_trades[pnl_col].astype(float).mean()) if win_count > 0 else 0.0
        avg_loss_float = abs(float(losing_trades[pnl_col].astype(float).mean())) if loss_count > 0 else 0.0
        profit_loss_ratio = avg_profit_float / avg_loss_float if avg_loss_float > 0 else float('inf')
        expected_value = win_rate * avg_profit_float - (1 - win_rate) * avg_loss_float

        stats = {
            '总交易次数': total_trades, '盈利交易次数': win_count, '亏损交易次数': loss_count,
            '总盈亏': total_pnl, # Keep as Decimal
            '最大盈利': max_profit, # Keep as Decimal
            '最大亏损': max_loss, # Keep as Decimal
            '平均盈亏': Decimal(avg_pnl_float), # Convert back for display consistency
            '盈亏标准差': Decimal(std_dev_pnl_float), # Convert back
            '买入次数': len(df[df[side_col].str.upper() == 'BUY']),
            '卖出次数': len(df[df[side_col].str.upper() == 'SELL']),
            '胜率': win_rate, # float
            '盈亏比': profit_loss_ratio, # float
            '期望值': expected_value, # float
        }
        return stats
    except Exception as e: st.error(f"生成交易统计时出错: {str(e)}"); import traceback; st.code(traceback.format_exc(), language="python"); return None

def display_trade_statistics(stats, symbol=None):
    """显示交易统计信息"""
    if not stats: st.info("没有足够的数据生成统计信息"); return
    title = f"{symbol} 交易统计" if symbol else "整体交易统计"
    st.write(f"### {title}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总交易次数", f"{stats['总交易次数']}")
        st.metric("胜率", f"{stats['胜率']:.2%}")
        # FIX 1: Use math.isfinite for float profit_loss_ratio
        st.metric("盈亏比", f"{stats['盈亏比']:.2f}" if math.isfinite(stats['盈亏比']) else "无限大")
    with col2:
        st.metric("总盈亏", f"{stats['总盈亏']:.4f}")
        st.metric("最大盈利", f"{stats['最大盈利']:.4f}")
        st.metric("最大亏损", f"{stats['最大亏损']:.4f}")
    with col3:
        st.metric("平均盈亏/每笔", f"{stats['平均盈亏']:.4f}")
        st.metric("盈利交易次数", f"{stats['盈利交易次数']}")
        st.metric("亏损交易次数", f"{stats['亏损交易次数']}")