# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair, informative,
                                stoploss_from_absolute, stoploss_from_open)

from freqtrade.exchange import timeframe_to_seconds, timeframe_to_minutes

import time as t
import datetime
from freqtrade.persistence import Trade
from pycoin.strategies.indicator_based_strategies.KalmanFilterStrategy import Kalmanfilter
from pycoin import Utils



# This class is a sample. Feel free to customize it.
class KalmanFilterStrategy(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "60": 0.01,
    #     "30": 0.02,
    #     "0": 0.04
    # }

    stoploss = -0.005

    # Trailing stoploss
    trailing_stop = False
    use_custom_stoploss = True 

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    
    timeframe_in_min = timeframe_to_minutes(timeframe)
    timeframe_in_sec = timeframe_to_seconds(timeframe)
    ignore_buying_expired_candle_after = timeframe_in_sec*2 # in seconds

    # Hyperoptable and strategy params
    
    # kalman filter object
    # kf = Kalmanfilter(observation_covariance = 0.05, #Variance of the observations (larger values allow for more noise).
    #                   transition_covariance = 0.01) #Variance of the state transitions (smaller values make the filter smoother).
    
    MAX_PctPriceDist_From_Signal = 0.01
    

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 5

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # plot_config = {
    #     'main_plot': {
    #         'tema': {},
    #         'sar': {'color': 'white'},
    #     },
    #     'subplots': {
    #         "MACD": {
    #             'macd': {'color': 'blue'},
    #             'macdsignal': {'color': 'orange'},
    #         },
    #         "RSI": {
    #             'rsi': {'color': 'red'},
    #         }
    #     }
    # }

    def bot_start(self, **kwargs):
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        t0 = t.time()
        symbol = metadata["pair"]
        if "enter_long" not in dataframe.columns: dataframe["enter_long"] = 0
        if "exit_long" not in dataframe.columns: dataframe["exit_long"] = 0
        
        # generating "Position_side"(0, 1 and -1 values) and "Kalman" columns 
        kf = Kalmanfilter(observation_covariance = 0.05, transition_covariance = 0.01)
        dataframe = kf.generate_signal(dataframe, highs_order = 25,
                                       lows_order = 25, filter_column = "close")
        dataframe = kf.generate_signal_range(dataframe, "Kalman", 0.01, 1)
        
        t_end = t.time()
        delta_time = t_end - t0
        print(f"\npopulation ended for {symbol} in {delta_time} secs\n")
        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        grp_long = dataframe.groupby("Position_side").get_group(1)["date"]
        last_longs = [str(dt) for dt in grp_long.dt.to_pydatetime()][-5:] # change n for other numbers of data
        long_log = f"LONG signal at Min, last {len(last_longs)} LONGs: {last_longs}"
        
        dataframe.loc[
            (   
                (dataframe["Position_side"] == 1) &
                (dataframe['volume'] > 0) 
            ), ['enter_long', 'enter_tag']] = (1, long_log)
        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        grp_short = pd.to_datetime(dataframe.groupby("Position_side").get_group(-1)["date"])    
        last_shorts = [str(dt) for dt in grp_short.dt.to_pydatetime()][-5:]
        short_log = f"SHORT signal at Max, last {len(last_shorts)} SHORTs: {last_shorts}"
        
        dataframe.loc[
            (   
             (dataframe["Position_side"] == -1) &
             (dataframe['volume'] > 0) 
            ), ['exit_long', 'exit_tag']] = (1, short_log)
        return dataframe
    
    
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) :
        
        # save_profits = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        # for target_profit in save_profits:
        #     if current_profit > target_profit: 
                
        
        # profit based SL
        ## always save 8% profit if profit>=10%
        if current_profit >= 0.10: return current_profit-0.08
        ## always save 6% profit if profit>=3%
        if current_profit >= 0.08: return current_profit-0.06
        ## always save 4% profit if profit>=6%
        if current_profit >= 0.06: return current_profit-0.04
        ## always save 2% profit if profit>=4%
        if current_profit >= 0.04: return current_profit-0.02
        ## always save 1% profit if profit>=2%
        if current_profit >= 0.02: return current_profit-0.01         
        if current_profit > 0.01: return current_profit-0.01
        return -1
      
        
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2
            },
            {
                "method": "MaxDrawdown",
                "lookback_period": 50,
                "trade_limit": 10,
                "stop_duration_candles": 10,
                "max_allowed_drawdown": 0.40
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 50,
                "trade_limit": 4,
                "stop_duration_candles": 5,
                "only_per_pair": True
            },
            {
                "method": "LowProfitPairs",
                "lookback_period": 24*60, # mins
                "trade_limit": 3,
                "stop_duration": 24*60,
                "required_profit": 0.015,
            }
        ]
        
        
