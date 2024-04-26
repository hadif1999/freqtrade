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
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from freqtrade.persistence import Trade


class PivotStrategy(IStrategy):
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
    timeframe = '15m'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "120": 0.02,
        "60": 0.04, # anytime after 60min reaches 1% profit exits position
        "30": 0.06, # anytime after 30min reaches 2% profit exits position
        "0": 0.08 # anytime reaches 4% profit exits position
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
    ignore_buying_expired_candle_after = timeframe_in_sec # in seconds

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 2

    # Strategy parameters
    # buy_rsi = IntParameter(10, 40, default=30, space="buy")
    # sell_rsi = IntParameter(60, 90, default=70, space="sell")
    PivotType:Literal["1w", '1M'] = "1w"
    pair_data = {}
    IGNORE_MEAN:bool = False
    TIMEOUT = dt.timedelta(days = 4)
    MIN_DELTA_TIME = dt.timedelta(hours = 0.5)
    MIN_C2_SIZE:float|None = 40
    MIN_C1_SIZE:float|None = None 
    ISC2_PRICE_TOLERANCE:float = 0.001
    ISC2_TIMEOUT:dt.timedelta = dt.timedelta(minutes = timeframe_in_min)
    
    

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

    
    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []
    
    
    @informative(PivotType)
    def populate_indicators_1w(self, dataframe:pd.DataFrame, metadata:dict):
        # df as dataframe containing pivot
        dataframe["mean"] = dataframe[["low", "high"]].mean(axis=1)
        dataframe["25%"] = dataframe[["low", "mean"]].mean(axis=1)
        dataframe["75%"] = dataframe[["mean", "high"]].mean(axis=1)
        return dataframe
        

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
        symbol = metadata["pair"]
        last_close = dataframe["close"].iloc[-1]
        
        # macd = ta.MACD(dataframe)
        # dataframe['macd'] = macd['macd']
        # dataframe['macdsignal'] = macd['macdsignal']
        # dataframe['macdhist'] = macd['macdhist']
        if "enter_long" not in dataframe.columns: dataframe["enter_long"] = 0
        if "exit_long" not in dataframe.columns: dataframe["exit_long"] = 0
        
        if symbol not in self.pair_data:
            self.pair_data[symbol] = {"Pivots":{},
                                      "init_LONG_SL": None, "init_LONG_TP": None,
                                      "init_SHORT_SL": None, "init_SHORT_TP": None}
        
        pivots = dataframe[[f"low_{self.PivotType}", f"25%_{self.PivotType}",
                    f"mean_{self.PivotType}", f"75%_{self.PivotType}",
                    f"high_{self.PivotType}"]].iloc[-1].to_dict()
        print(f"\n{last_close = }| {symbol = }| {pivots = }\n")
        return dataframe
    
    

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
        
        # getting last pivot
        pivots = dataframe[[f"low_{self.PivotType}", f"25%_{self.PivotType}",
                            f"mean_{self.PivotType}", f"75%_{self.PivotType}",
                            f"high_{self.PivotType}"]].iloc[-1].to_dict()
        
        self.pair_data[symbol]["Pivots"] = pivots # saving current pivots to pair data
                
        # finding all C1, C2 candles according to place of given price relative to pivots 
        # sorted by C2 date ([-1] is the newest)
        C1C2s = FindAllC1C2_Candles_FromPrice(Price=last_close, 
                                          Pivots_dict = pivots, df = dataframe,
                                          ignore_mean=False, timeout=self.TIMEOUT,
                                          min_time=self.MIN_DELTA_TIME,
                                          min_C2_Size=self.MIN_C2_SIZE,
                                          min_C1_size=self.MIN_C1_SIZE, 
                                          C1Type="bearish", C2Type="bullish")
        
        if C1C2s == []: return dataframe
        last_C1, last_C2 = C1C2s[-1]["C1"], C1C2s[-1]["C2"]
        C1_close, C2_close = last_C1["close"], last_C2["close"]
        
        print(f"{symbol = }, last LONG C1, C2-> {last_C1 = }, {last_C2 = }\n")
        equalTime_cond = last_close_time - last_C2["date"] <= self.ISC2_TIMEOUT
        equalPrice_cond = isEqual_with_tolerance_pct(last_close, C2_close, 
                                                     self.ISC2_PRICE_TOLERANCE)
        if equalPrice_cond and equalTime_cond:
            long_log = f"**LONG sig detected, {symbol = }, {C1_close = }|{C2_close = }"
            dataframe.loc[(dataframe["close"] == C2_close) & (dataframe["volume"] > 0), 
                         ['enter_long', 'enter_tag'] ] = (1, long_log)
            print(long_log)    
            self.pair_data[symbol]["init_LONG_SL"] = last_C2["low"]
            self.pair_data[symbol]["init_LONG_TP"] = pivots[f"mean_{self.PivotType}"]
            self.pair_data[symbol]["init_SHORT_SL"] = None
            self.pair_data[symbol]["init_SHORT_TP"] = None
            
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
        
        pivots = dataframe[[f"low_{self.PivotType}", f"25%_{self.PivotType}", 
                            f"mean_{self.PivotType}", f"75%_{self.PivotType}",
                            f"high_{self.PivotType}"]].iloc[-1].to_dict()
        self.pair_data[symbol]["Pivots"] = pivots
        
        C1C2s = FindAllC1C2_Candles_FromPrice(Price=last_close, 
                                          Pivots_dict = pivots, df = dataframe,
                                          ignore_mean=False, timeout=self.TIMEOUT,
                                          min_time=self.MIN_DELTA_TIME, 
                                          min_C2_Size=self.MIN_C2_SIZE,
                                          min_C1_size=self.MIN_C1_SIZE, 
                                          C1Type="bullish", C2Type="bearish")
        
        if C1C2s == []: return dataframe
        
        last_C1, last_C2 = C1C2s[-1]["C1"], C1C2s[-1]["C2"]
        C1_close, C2_close = last_C1["close"], last_C2["close"]
        
        print(f"{symbol = }, last SHORT C1, C2 -> {last_C1 = }, {last_C2 = }\n")
        
        equalTime_cond = last_close_time - last_C2["date"] <= self.ISC2_TIMEOUT
        equalPrice_cond = isEqual_with_tolerance_pct(last_close, C2_close, 
                                                     self.ISC2_PRICE_TOLERANCE)
        if equalPrice_cond and equalTime_cond:
            short_log = f"**SHORT sig detected, {symbol = }, {C1_close = }|{C2_close = }"
            dataframe.loc[(dataframe["close"] == C2_close) & (dataframe["volume"] > 0), 
                         ['exit_long', 'exit_tag'] ] = (1, short_log)
            print(short_log)  
            self.pair_data[symbol]["init_LONG_SL"] = None
            self.pair_data[symbol]["init_LONG_TP"] = None
            self.pair_data[symbol]["init_SHORT_SL"] = None   ### last_C2["high"]
            self.pair_data[symbol]["init_SHORT_TP"] = None   # pivots[f"mean_{self.PivotType}"]
        return dataframe
    
    
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) :
        
        init_SL = self.pair_data[pair]["init_LONG_SL"] or abs(self.stoploss)
        init_SL_pct = stoploss_from_absolute(init_SL, current_rate, 
                                            trade.is_short, trade.leverage)
        
        pivots = self.pair_data[pair]["Pivots"]
        nearPivots_open = _find_nearest_pivots(trade.open_rate, pivots)
        nearPivots_current = _find_nearest_pivots(current_rate, pivots)
        if nearPivots_open.keys() == nearPivots_current.keys():
            return min(init_SL_pct, abs(self.stoploss))
        else: 
            minPivot_near = min(nearPivots_current.values())
            return stoploss_from_absolute(minPivot_near, current_rate, 
                                          trade.is_short, trade.leverage)
                
                
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                            rate: float, time_in_force: str, exit_reason: str,
                            current_time: datetime, **kwargs) -> bool:
        
        self.pair_data[pair]["init_LONG_SL"] = None
        self.pair_data[pair]["init_LONG_TP"] = None
        return True
    


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
                "max_allowed_drawdown": 0.2
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
    
######################################### utils functions #####################

from typing import Literal

# check if last pivots are long position pivots or not
def _findSide_fromPivots(close_price:float,
                         pivots_dict: dict[str,float]) -> Literal['SHORT', 'LONG']:
    """find position side from input price according to it's nearest pivots 

    Raises:
        ValueError: _description_

    Returns:
        "LONG" or "SHORT"
    """        
    return "SHORT" if close_price >= pivots_dict[f"mean_{PivotStrategy.PivotType}"] else "LONG" 




def getBulish_CrossedPrice(df: pd.DataFrame, Price:float) -> pd.DataFrame:
    cond = (df["close"] > Price) & (df["open"] < Price)
    return df.loc[cond, :]


def getBearish_CrossedPrice(df: pd.DataFrame, Price: float) -> pd.DataFrame:
    cond = (df["close"] < Price) & (df["open"] > Price)
    return df.loc[cond, :]



def _find_nearest_pivots(price:float, pivots_dict: dict[str, float]):
        """find one/two nearest pivots to a price 

        Args:
            price (float): price 

        Returns:
            tuple[float, float]| None, float | float, None: pivots
        """                
        lower_pivots = {}
        upper_pivots = {}
        for name, val in pivots_dict.items():
            if price >= val: lower_pivots[name] = val
            elif price < val: upper_pivots[name] = val

        upper_pivot = [min(upper_pivots.items(), key = lambda x:x[1])] if upper_pivots!={} else {}  
        lower_pivot = [max(lower_pivots.items(), key = lambda x:x[1])] if lower_pivots!={} else {} 
        return {**dict(lower_pivot), **dict(upper_pivot)}



def _GetCrossedPivot_Candles(Price: float, *, 
                             pivots_dict:dict[str, float],
                             df:pd.DataFrame, 
                             candle_type:CandleType = "bullish"):
    """finds last 'bullish'/'bearish' candle that crossed a pivot. 

    Args:
        dataframe (pd.DataFrame, optional): finds candle on this df. Defaults to None.
        candle_type (str, optional): 'bullish' or 'bearish'. Defaults to "bullish".

    Raises:
        ValueError: _description_

    Returns:
        candle_indice, crossed pivot
    """        
    
    df_ = df.copy()
    
    pivots_dict = _find_nearest_pivots(price = Price, pivots_dict = pivots_dict)
    
    match candle_type.lower():
        case "bullish":
            return {pivot_name: getBulish_CrossedPrice(df_, pivot) 
                    for pivot_name, pivot in pivots_dict.items()}
            
        case "bearish":
            return {pivot_name: getBearish_CrossedPrice(df_, pivot)
                    for pivot_name, pivot in pivots_dict.items()} 
            
        case _ : 
            raise ValueError("candle_type can be 'bullish' or 'bearish'")
        
        
        

def _FindAllC1C2_Candles_FromPrice( Price: float, *, 
                                    Pivots_dict:dict[str ,float],
                                    df: pd.DataFrame = pd.DataFrame(),
                                    C1_Type:CandleType = "bearish",
                                    C2_Type:CandleType = "bullish",
                                    ignore_mean: bool = False):
        
    assert C1_Type != C2_Type, "C1 and C2 candle type must be diffrent"
    cols = ['open', 'high', 'low', 'close', 'volume', "date"]
    df_ = df[cols].copy()
    if "datetime" not in df_.index.dtype.name: 
        df_.set_index("date", inplace = True, drop = False) 
        
    C1_dict = _GetCrossedPivot_Candles(Price=Price, pivots_dict=Pivots_dict,
                                       df = df_, candle_type = C1_Type)
    
    C2_dict = _GetCrossedPivot_Candles(Price=Price, pivots_dict=Pivots_dict,
                                       df = df_, candle_type = C2_Type)
    
    assert C1_dict.keys() == C2_dict.keys(), "pivotLevels found must be same!"
    pivot_levelNames = C1_dict.keys()
    c1, c2 = "_C1", "_C2"
    C1_colNames, C2_colNames = [col+c1 for col in cols], [col+c2 for col in cols]
    # getting C1,C2 pair
    C1C2_candles = {}
    for levelName in pivot_levelNames:
        if ignore_mean and "mean" in levelName.lower(): continue
        C1_df, C2_df = C1_dict[levelName], C2_dict[levelName]
        # adding suffix to C1,C2 col names to identify them easily
        C1_df = add_to_ColumnNames(C1_df.copy(), suffix = c1)
        C2_df = add_to_ColumnNames(C2_df.copy(), suffix = c2)
        ## first concat to add indices
        C1C2_df = pd.concat([C1_df, C2_df], axis = 1)
        # shifting C1 part to align indexes and removing empty rows
        shifted_C1_df = C1C2_df[C1_colNames].shift(1)
        C1C2_df = pd.concat([shifted_C1_df, C2_df], axis = 1).dropna().copy()
        ## reverting column names to their initial names
        _C1_df, _C2_df = C1C2_df[C1_colNames], C1C2_df[C2_colNames]
        _C1_df = remove_from_ColumnNames(_C1_df, suffix = c1)
        _C2_df = remove_from_ColumnNames(_C2_df, suffix = c2)
        # making a list of dicts with C1 and C2 keys
        C1C2_candles[levelName] = [{"C1":C1[1].to_dict(), "C2":C2[1].to_dict()} 
                                    for C1, C2 in zip(_C1_df.iterrows(), _C2_df.iterrows())]
    # putting all levels C1, C2s in a single list
    all_C1C2_candles = []
    for C1C2s in C1C2_candles.values(): all_C1C2_candles += C1C2s
    all_C1C2_candles = sorted(all_C1C2_candles, key = lambda x: x["C2"]["date"])
    return all_C1C2_candles



def FindAllC1C2_Candles_FromPrice(  Price: float, *, 
                                    Pivots_dict:dict[str ,float],
                                    df: pd.DataFrame = pd.DataFrame(),
                                    ignore_mean: bool = False, 
                                    C1Type:CandleType = "bearish", 
                                    C2Type:CandleType = "bullish",
                                    timeout:dt.timedelta = dt.timedelta(days=3),
                                    min_time :dt.timedelta = dt.timedelta(hours=0.5),
                                    min_C2_Size:int|None = 100, 
                                    min_C1_size:int|None = None):
    
    C1C2s_raw = _FindAllC1C2_Candles_FromPrice(Price=Price, Pivots_dict=Pivots_dict,
                                            df=df, C1_Type=C1Type, C2_Type=C2Type,
                                            ignore_mean=ignore_mean)    
    all_C1C2_candles = C1C2s_raw.copy()
    ##### adding filter to found C1,C2s
    all_C1C2_candles = C1C2_time_filter(all_C1C2_candles, timeout = timeout,  # adding time filter
                                        min_time=min_time)
    if min_C2_Size: all_C1C2_candles = C1C2_size_filter(all_C1C2_candles, # C2 size
                                                        CandleType="C2", 
                                                        minSize=min_C2_Size)
    
    if min_C1_size: all_C1C2_candles = C1C2_size_filter(all_C1C2_candles, # C1 size
                                                        CandleType="C1",
                                                        minSize=min_C1_size)
    # removing repeated C1C2s
    for C1C2 in all_C1C2_candles: 
        while all_C1C2_candles.count(C1C2) > 1: all_C1C2_candles.remove(C1C2)
    # sorting C1,C2 s by C2 date
    all_C1C2_candles = sorted(all_C1C2_candles, key=lambda x: x["C2"]["date"] )
    return all_C1C2_candles
    



def C1C2_time_filter(C1C2s:list[dict[str,dict]], 
                    timeout:dt.timedelta = dt.timedelta(days = 4), 
                    min_time:dt.timedelta = dt.timedelta(hours = 0.5)):
    assert timeout > min_time, "timout must be longer than min_time"
    all_C1C2s = []
    for C1C2 in C1C2s: 
        delta_t = C1C2["C2"]["date"] - C1C2["C1"]["date"]        
        cond = min_time < delta_t <= timeout
        if min_time < delta_t <= timeout: 
            all_C1C2s.append(C1C2)        
    return all_C1C2s
    
    
def C1C2_size_filter(C1C2s: list[dict[str,dict]], 
                     CandleType: C1C2Type = "C2", 
                     minSize:float = 100):
    _C1C2s = C1C2s.copy()
    return [C1C2 for C1C2 in _C1C2s 
            if abs(C1C2[CandleType]["close"] - C1C2[CandleType]["open"]) >= minSize]   

    
def add_to_ColumnNames(dataframe: pd.DataFrame, *, prefix:str = '', suffix:str = ''):
    return dataframe.rename(columns={col: prefix+col+suffix for col in dataframe.columns}).copy()


def remove_from_ColumnNames(dataframe: pd.DataFrame, *, prefix:str = '', suffix:str = ''):
    return dataframe.rename(columns={col: col.removeprefix(prefix).removesuffix(suffix) 
                              for col in dataframe.columns}).copy()
    
    
def isEqual_with_tolerance_pct(val1: float, val2: float, tol: float = 0.001):
    val2_upper = val2 + tol * val2
    val2_lower = val2 - tol * val2
    return val1 > val2_lower and val1 < val2_upper
    
    
        
def _GetLatestCrossedPivot_candle_from_LastClose(*, 
                                                dataframe: pd.DataFrame = pd.DataFrame(),
                                                pivots_dict: dict[str,float],
                                                candle_type: CandleType = "bullish"):
    
    """get the latest pivot crossed pivot candle relative to last close

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """        
    crossed_candles_dict = _GetCrossedPivot_Candles(df = dataframe, 
                                                    candle_type = candle_type,
                                                    pivots_dict = pivots_dict)
    
    lastCrossed_candle = max( [df.iloc[-1] for levelName, df in crossed_candles_dict.items()
                                if "mean" not in levelName and not df.empty], 
                                key = lambda x: x.date, default = pd.Series() ) 
    return lastCrossed_candle
                    
###########################################################              
                    
                    
                    
