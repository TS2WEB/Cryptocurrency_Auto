import ccxt
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import pytz

# 创建OKEx交易所实例
exchange = ccxt.okx()

def fetch_market_data(symbol, timeframe, limit=100):
    """
    获取指定交易对的K线数据，并计算技术指标。
    包含：
    - MACD(12,26,9)
    - RSI-7
    - MA系列指标
    - 成交量MA
    """
    try:
        # 获取额外的K线数据以确保指标计算准确性
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit + 50)
        if not ohlcv or len(ohlcv) < limit:
            print(f"警告：{symbol} 在 {timeframe} 时间框架中的数据不足")
            return None
            
        # 创建DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 验证数据时间戳是否最新
        current_time = exchange.milliseconds()
        last_candle_time = df['timestamp'].iloc[-1]
        max_delay = {
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000
        }
        
        if (current_time - last_candle_time) > max_delay[timeframe]:
            print(f"警告: {symbol} {timeframe} 数据不是最新的")
            return None

        # 数据类型转换和验证
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                print(f"警告：{symbol} 在 {timeframe} 时间框架中存在无效的{col}数据")
                return None

        # 计算技术指标
        try:
            # MACD - 使用标准参数(12,26,9)
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            macd.columns = ['MACD', 'MACD_signal', 'MACD_hist']
            df = pd.concat([df, macd], axis=1)

            # RSI - 使用7周期
            df['RSI_7'] = ta.rsi(df['close'], length=7)
            df['RSI_7_prev'] = df['RSI_7'].shift(1)

            # 移动平均线
            df['MA5'] = ta.sma(df['close'], length=5)
            df['MA10'] = ta.sma(df['close'], length=10)
            df['MA20'] = ta.sma(df['close'], length=20)

            # 成交量移动平均线
            df['volume_MA5'] = ta.sma(df['volume'], length=5)
            df['volume_MA10'] = ta.sma(df['volume'], length=10)

            # 计算当前K线成交量与前5根K线平均成交量的比值
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=5).mean().shift(1)

        except Exception as e:
            print(f"计算技术指标时出错 {symbol} {timeframe}: {str(e)}")
            return None

        # 验证关键指标是否存在
        required_columns = ['close', 'RSI_7', 'MACD_hist', 'MA5', 'MA10', 'MA20', 
                          'volume_MA5', 'volume_MA10', 'volume_ratio']
        
        if not all(col in df.columns for col in required_columns):
            print(f"警告: {timeframe} 时间框架缺少必要的技术指标")
            return None

        # 删除空值并验证数据完整性
        df = df.dropna()
        if len(df) < limit:
            print(f"警告：{symbol} 在 {timeframe} 时间框架中的有效数据不足")
            return None

        return df.tail(limit)  # 只返回需要的数据点数

    except Exception as e:
        print(f"获取数据时出错 {symbol} {timeframe}: {str(e)}")
        return None

def check_1h_conditions(df):
    """检查1小时时间框架的条件"""
    if df is None or df.empty:
        return False

    try:
        latest = df.iloc[-1]
        
        # 计算价格与MA20的偏离百分比
        ma20_deviation = (latest['close'] - latest['MA20']) / latest['MA20'] * 100

        return (
            latest['close'] > latest['MA20'] and
            latest['MA10'] > latest['MA20'] and
            latest['MACD_hist'] > 0 and
            ma20_deviation < 2 and
            latest['volume_MA5'] > latest['volume_MA10']
        )

    except Exception as e:
        print(f"检查1小时条件时出错: {str(e)}")
        return False

def check_15m_conditions(df):
    """检查15分钟时间框架的条件"""
    if df is None or df.empty:
        return False

    try:
        latest = df.iloc[-1]
        
        return (
            latest['MA5'] > latest['MA10'] and
            40 < latest['RSI_7'] < 70 and
            latest['volume_ratio'] > 1.5 and
            latest['volume_MA5'] > latest['volume_MA10']
        )

    except Exception as e:
        print(f"检查15分钟条件时出错: {str(e)}")
        return False

def check_5m_conditions(df):
    """检查5分钟时间框架的条件"""
    if df is None or df.empty:
        return False

    try:
        latest = df.iloc[-1]
        
        return (
            latest['RSI_7'] > latest['RSI_7_prev'] and
            latest['MACD_hist'] > 0
        )

    except Exception as e:
        print(f"检查5分钟条件时出错: {str(e)}")
        return False

def filter_by_conditions(symbol):
    """
    检查指定交易对在所有时间框架下是否符合筛选条件。
    要求所有时间周期都满足各自的条件。
    """
    timeframes = {
        '1h': check_1h_conditions,
        '15m': check_15m_conditions,
        '5m': check_5m_conditions
    }
    
    results = {}
    all_conditions_met = True

    for timeframe, check_func in timeframes.items():
        df = fetch_market_data(symbol, timeframe)
        if df is not None:
            is_satisfied = check_func(df)
            results[timeframe] = is_satisfied
            if not is_satisfied:
                all_conditions_met = False
        else:
            all_conditions_met = False
            results[timeframe] = False

    print(f"\n{symbol} 各时间框架检查结果:")
    for tf, result in results.items():
        print(f"{tf}: {'满足' if result else '不满足'}")

    return all_conditions_met

def get_top_volume_perpetual(top_n=195):
    """
    获取成交量前top_n的U本位永续合约。
    """
    try:
        markets = exchange.load_markets()
        perpetual_pairs = [
            symbol for symbol, market in markets.items()
            if market['type'] == 'swap' and market['quote'] == 'USDT' and market['active']
        ]

        tickers = exchange.fetch_tickers(perpetual_pairs)
        market_data = [{
            'symbol': symbol,
            'base': markets[symbol]['base'],
            'last_price': tickers[symbol]['last']
        } for symbol in perpetual_pairs]

        df = pd.DataFrame(market_data).head(top_n)
        current_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')

        print(f"\n获取时间（北京时间）: {current_time}")
        print(f"\n成交量前 {top_n} 的U本位永续合约:")
        print(df[['symbol', 'base', 'last_price']])

        return df, current_time

    except Exception as e:
        print(f"获取市场数据时出错: {str(e)}")
        return None, None

def main():
    """
    主程序入口：
    1. 获取高成交量的永续合约
    2. 对每个合约进行多时间周期技术分析
    3. 输出并保存满足条件的交易对
    """
    try:
        df, current_time = get_top_volume_perpetual()
        if df is not None and not df.empty:
            # 筛选符合条件的交易对
            positive_symbols = [symbol for symbol in df['symbol'] if filter_by_conditions(symbol)]

            # 打印和保存结果
            if positive_symbols:
                print("\n同时满足所有筛选条件的交易对:")
                for symbol in positive_symbols:
                    print(symbol)

                # 保存结果
                result_df = pd.DataFrame({
                    'symbol': positive_symbols,
                    'beijing_time': current_time
                })
                filename = f"screened_symbols_{int(datetime.now().timestamp())}.csv"
                result_df.to_csv(filename, index=False)
                print(f"\n筛选结果已保存至: {filename}")
            else:
                print("\n没有交易对满足所有筛选条件。")
    except Exception as e:
        print(f"运行过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()
