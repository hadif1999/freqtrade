# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime 
import datetime as dt
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair, informative,
                                stoploss_from_absolute, stoploss_from_open)

from freqtrade.exchange import timeframe_to_seconds, timeframe_to_minutes
from typing import Literal
import os

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from freqtrade.persistence import Trade
from typing import Literal
import time as t

from pycoin.strategies.level_based_strategies.fract_levels_strategy import Fract_Levels
from pycoin import Utils


class FractLevels(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 1
    

    # Optimal timeframe for the strategy.
    timeframe = '30m'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "120": 0.02,
    #     "60": 0.04, # anytime after 60min reaches 1% profit exits position
    #     "30": 0.06, # anytime after 30min reaches 2% profit exits position
    #     "0": 0.08 # anytime reaches 8% profit exits position
    # }
    
    minimal_roi = {
        "0": 0.936,
        "5271": 0.332,
        "18147": 0.086,
        "48152": 0
    }

    # STOPLOSS
    # src: https://www.freqtrade.io/en/stable/stoploss/#trailing-stop-loss-custom-positive-loss
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.05
    # Trailing stoploss
    # trailing_stop = True
    # trailing_only_offset_is_reached = True # keep inital stoploss untill reaches to offset
    # trailing_stop_positive = 0.02 # stoploss after reaching to offset
    # trailing_stop_positive_offset = 0.03  # Disabled / not configured
    
    # https://www.freqtrade.io/en/stable/strategy-callbacks
    use_custom_stoploss = True 

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    timeframe_in_min = timeframe_to_minutes(timeframe)
    timeframe_in_sec = timeframe_to_seconds(timeframe)
    ignore_buying_expired_candle_after = timeframe_in_sec*2 # in seconds

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 2

    ######################## strategy variables ############
    pair_data:dict[str, dict[str, float]] = {} 
    init_SL:dict[str, dict[str, float]] = {} # init_SL[symbol] = {"LONG":float, "SHORT":float}
    SL_counter:dict[str, int] = {} # SL_counter[symbol] = int
    
    ISC2_PRICE_TOLERANCE:float = 0.001
    MAX_SL_COUNT = 5 # starts to count each time crossed SL till reaches this param then exit
    ISC2_TIMEOUT:dt.timedelta = dt.timedelta(minutes = timeframe_in_min) # max time dist between signal and last close
    ########################
    
    ############################## strategy configs #############
    C1C2_long_conf = dict(
            C1_Type = "bearish", C2_Type="bullish", 
            timeout = dt.timedelta(days=1e4), # max timeout between C1, C2 candles
            min_delta_time=dt.timedelta(hours = 2), # min time dist between C1, C2
            betweenCandles_maxPeakDist= 0.0015, # min dist of candles between C1, C2 peak
            minC2_size = 10,
            ignore_timeFilter=False, 
            ignore_C1C2_sizeFilter=False, 
            ignore_lowestlowDist_filter=True, 
            ignore_HighLow = False) # add C1, C2 if their shadows had this condition
    
    C1C2_short_conf = C1C2_long_conf.copy()
    C1C2_short_conf.update(dict(C1_Type = "bullish", C2_Type="bearish"))
    
    DataExchange = os.getenv("DATA_EXCHANGE", "binance")
    FractLevels_conf = dict(
                    timeframe = '1w', # strategy main timeframe (not used here and can be ignored)
                    data_exchange = DataExchange, # exchange to eval fract levels from
                    start_time = dt.datetime(2018, 1, 1), 
                    limit = 1000,
                    title_col_names = False, 
                    datetime_index = False)
    
    FractLevels_finder_conf = dict(
                        method="fracts",# use past 'fracts' or 'pivots' or both
                        candle_ranges = None, # window size for evaluating high and lows
                        accuracy_pct=1e-7, # a tolerance to specify how much accurate touches must be 
                        min_occurred=3, # number of touches
                        find_onIntervals=["1d"], # intervals to find fracts on
                        min_FractsDist_Pct=0.02 ) # min dist between fracts 
    ####################################################
    
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
    
    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'tema': {},
                'sar': {'color': 'white'},
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    'macd': {'color': 'blue'},
                    'macdsignal': {'color': 'orange'},
                },
                "RSI": {
                    'rsi': {'color': 'red'},
                }
            }
        }

    
    # def informative_pairs(self):
    #     """
    #     Define additional, informative pair/interval combinations to be cached from the exchange.
    #     These pair/interval combinations are non-tradeable, unless they are part
    #     of the whitelist as well.
    #     For more information, please consult the documentation
    #     :return: List of tuples in the format (pair, interval)
    #         Sample: return [("ETH/USDT", "5m"),
    #                         ("BTC/USDT", "15m"),
    #                         ]
    #     """
    #     return []
    

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        t0 = t.time()
        symbol = metadata["pair"]
        last_close = dataframe["close"].iloc[-1]
        
        # macd = ta.MACD(dataframe)
        # dataframe['macd'] = macd['macd']
        # dataframe['macdsignal'] = macd['macdsignal']
        # dataframe['macdhist'] = macd['macdhist']
        
        # initialize dataframe and pair data if they don't exist
        if "enter_long" not in dataframe.columns: dataframe["enter_long"] = 0
        if "exit_long" not in dataframe.columns: dataframe["exit_long"] = 0
        self.initialize_PairData(symbol)
        
        df_ = dataframe.copy()
        df_ = Utils.to_standard_OHLCV_dataframe(df_)
        C1C2_long, C1C2_short, levels = FractLevels.last_LongShort_C1C2(df_,
                                                                        symbol,
                                                                        last_close)
        self.pair_data[symbol] = {
                                "LAST_C1C2_LONG": C1C2_long,
                                "LAST_C1C2_SHORT": C1C2_short,
                                "levels": levels}
        t_end = t.time()
        delta_time = t_end - t0
        print(f"\npopulation ended for {symbol} in {delta_time} secs\n")
        return dataframe
    
    
    def initialize_PairData(self, symbol:str):
        
        if symbol not in self.pair_data.keys():
            self.pair_data[symbol] = {
                            "LAST_C1C2_LONG": {},
                            "LAST_C1C2_SHORT": {},
                            "levels": []}
        
        if symbol not in self.init_SL.keys():
            self.init_SL[symbol] = {"LONG": None, "SHORT": None}
        
        if symbol not in self.SL_counter.keys():
            self.SL_counter[symbol] = 0
        
    
    def last_LongShort_C1C2(dataframe:pd.DataFrame,
                            symbol:str, lastClose: float):
        
        fracts = Fract_Levels(symbol = symbol, **FractLevels.FractLevels_conf)
        levels = fracts.eval_fract_levels(**FractLevels.FractLevels_finder_conf, 
                                          rel_to_Price = lastClose)
        C1C2s_long = fracts.FindAllC1C2_Candles_FromPrice(Price = lastClose, df = dataframe,
                                                          **FractLevels.C1C2_long_conf)
        C1C2s_short = fracts.FindAllC1C2_Candles_FromPrice(Price = lastClose, df = dataframe,
                                                          **FractLevels.C1C2_short_conf)
        last_C1C2_long = None if C1C2s_long == [] else C1C2s_long[-1]
        last_C1C2_short = None if C1C2s_short == [] else C1C2s_short[-1]
        return last_C1C2_long, last_C1C2_short, levels
    

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        last_close = dataframe["close"].iloc[-1]
        symbol = metadata["pair"]
        last_close_time = dataframe["date"].iloc[-1]
                
        # finding last C1, C2 candles according to place of given price relative to fracts
        # sorted by C2 date ([-1] is the newest)
        lastC1C2_LONG = self.pair_data[symbol]["LAST_C1C2_LONG"]
        if lastC1C2_LONG == None: return dataframe
        last_C1, last_C2 = lastC1C2_LONG["C1"], lastC1C2_LONG["C2"]
        crossed_level = lastC1C2_LONG["level"]
        C1_close, C2_close = last_C1["Close"], last_C2["Close"]
        
        print(f"""\n{symbol = }, last LONG C1,C2-> {last_C1 = }, {last_C2 = }
              crossed at {crossed_level}\n""")
        
        equalTime_cond = last_close_time - last_C2["Datetime"] <= self.ISC2_TIMEOUT
        equalPrice_cond = isEqual_with_tolerance_pct(last_close, C2_close, 
                                                     self.ISC2_PRICE_TOLERANCE)
        if equalPrice_cond and equalTime_cond:
            long_log = f"""**LONG sig detected,{symbol = }, {last_C1["Close"] = },
            {last_C1["Datetime"] = }| {last_C2["Close"] = }, {last_C2["Datetime"] = }
            {crossed_level = }"""
            dataframe.loc[(dataframe["close"] == last_close) & (dataframe["volume"] > 0), 
                         ['enter_long', 'enter_tag'] ] = (1, long_log)
            print(long_log)  
            self.dp.send_msg(long_log)
            self.init_SL[symbol]["LONG"] = last_C2["Low"]
        return dataframe



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        last_close = dataframe["close"].iloc[-1]
        symbol = metadata["pair"]
        last_close_time = dataframe["date"].iloc[-1]

        lastC1C2_SHORT = self.pair_data[symbol]["LAST_C1C2_SHORT"]
        if lastC1C2_SHORT==None: return dataframe
        last_C1, last_C2 = lastC1C2_SHORT["C1"], lastC1C2_SHORT["C2"]
        crossed_level = lastC1C2_SHORT["level"]
        C1_close, C2_close = last_C1["Close"], last_C2["Close"]
        
        print(f"""\n{symbol = }, last SHORT C1,C2-> {last_C1 = }, {last_C2 = }
               crossed at {crossed_level}\n""")
        equalTime_cond = last_close_time - last_C2["Datetime"] <= self.ISC2_TIMEOUT
        equalPrice_cond = isEqual_with_tolerance_pct(last_close, C2_close, 
                                                     self.ISC2_PRICE_TOLERANCE)
        if equalPrice_cond and equalTime_cond:
            short_log = f"""**SHORT sig detected, {symbol = }, {last_C1["Close"] = },
            {last_C1["Datetime"] = }|{last_C2["Close"] = }, {last_C2["Datetime"] = },
            {crossed_level = }"""
            dataframe.loc[(dataframe["close"] == last_close) & (dataframe["volume"] > 0), 
                         ['exit_long', 'exit_tag'] ] = (1, short_log)
            print(short_log)  
            self.dp.send_msg(short_log)
            self.init_SL[symbol]["SHORT"] = last_C2["High"]            
        return dataframe
    
    
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) :
        
        is_long, is_short = not trade.is_short, trade.is_short
        
        # profit based SL
        ## always save 6% profit if profit>=3%
        if current_profit >= 0.06: return current_profit-0.03
        ## always save 2% profit if profit>=4%
        if current_profit >= 0.04: return current_profit-0.02
        ## always save 1% profit if profit>=2%
        if current_profit >= 0.02: return current_profit-0.01 
        
        # initial SL will be evaluated here
        init_SL = self.init_SL[pair].get("SHORT" if is_short else "LONG")
        if init_SL is None: init_SL = abs(self.stoploss) 
        else: 
            init_SL_pct = stoploss_from_absolute(init_SL, current_rate, 
                                                 is_short, trade.leverage)
            init_SL = min(abs(init_SL_pct), abs(self.stoploss))  
            init_SL = abs(self.stoploss) # delete this if you don't want const init SL
        
        lastC1C2 = self.pair_data[pair]["LAST_C1C2_SHORT" if is_short 
                                        else "LAST_C1C2_LONG"]
        crossed_level = lastC1C2["level"] 
        levels = self.pair_data[pair]["levels"]
        fracts_obj = Fract_Levels(symbol = pair, **FractLevels.FractLevels_conf)
        fracts_obj.fracts = levels
        nearFracts_current:list = fracts_obj._find_nearest_FractLevels(current_rate)
        nearFracts_open:list = fracts_obj._find_nearest_FractLevels(trade.open_rate)
        isTotallyDifferentFracts = all([True for fract in nearFracts_current 
                                 if fract not in nearFracts_open])
        meanNearFracts = np.mean(nearFracts_current)
        
        # SL is C2.low(for LONG) if price is between initial fracts
        if sorted(nearFracts_current) == sorted(nearFracts_open): 
            return init_SL
        
        # defining SL when all current near fracts are diffrent with opened ones 
        elif isTotallyDifferentFracts: 
            self.init_SL[pair]["SHORT"] = self.init_SL[pair]["LONG"] = None
            # for LONG side if current price above fracts mean put SL on mean
            # else put SL on smaller near fract
            if is_long: # is LONG
                if current_rate >= meanNearFracts: 
                    new_SL = meanNearFracts
                else: 
                    new_SL = min(nearFracts_current)
            else:
                if current_rate <= meanNearFracts:
                    new_SL = meanNearFracts
                else: 
                    new_SL = max(nearFracts_current)
                    
            new_SL_pct = stoploss_from_absolute(new_SL, current_rate,
                                                trade.is_short, trade.leverage)
            return new_SL_pct
        
        else: # if there is only 1 same fract, change SL to mean(fracts)
            self.init_SL[pair]["SHORT"] = self.init_SL[pair]["LONG"] = None
            if is_long: # is LONG
                if current_rate >= meanNearFracts: 
                    new_SL = meanNearFracts
                else: 
                    return -1 # keep the previous SL
            else:
                if current_rate <= meanNearFracts:
                    new_SL = meanNearFracts
                else: 
                    return -1 # keep the previous SL
                
            new_SL_pct = stoploss_from_absolute(new_SL, current_rate,
                                                is_short, trade.leverage)
            return new_SL_pct
        
        
        
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 50,
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
                "lookback_period_candles": 100,
                "trade_limit": 8,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]
        
        
########################################### dataTypes ###################
CandleType = Literal["bullish", "bearish"]
SideType = Literal["LONG", "SHORT"]
C1C2Type = Literal["C1", "C2"]
########################################################################
    
################################ utils #####################
def isEqual_with_tolerance_pct(val1: float, val2: float, tol: float = 0.001):
    val2_upper = val2 + tol * val2
    val2_lower = val2 - tol * val2
    return val2_lower <= val1 <= val2_upper
##########################################################
                    
