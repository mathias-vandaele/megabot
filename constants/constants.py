"""
author : Mathias Vandaele
"""
import os

from binance.client import Client


class Constants:

    api_key = os.environ.get('binance_api')
    api_secret = os.environ.get('binance_secret')
    kline_interval = Client.KLINE_INTERVAL_1HOUR

    # Timestamp when the candle is opened
    CANDLE_OPEN_TIMESTAMP = 0
    # Price when the candle is opened
    CANDLE_OPEN = 1
    # Highest asset price on the current candle
    CANDLE_HIGH = 2
    # Lowest asset price on the current candle
    CANDLE_LOW = 3
    # Price when the candle is closed
    CANDLE_CLOSE = 4
    # Volume of buy/sell on the current candle (ex : VOLUME in BTC if BTCUSDT selected)
    VOLUME = 5
    # Timestamp when the candle is closed
    CANDLE_CLOSE_TIMESTAMP = 6
    # Volume of buy/sell in quote asset volume (ex : VOLUME in USDT if BTCUSDT selected)
    QUOTE_VOLUME = 7
    # Number of trade made on the current candle
    TRADE_NUMBER = 8
    # Taker buy volume on the current candle (ex : TAKER BUY VOLUME in BTC if BTCUSDT selected)
    # meaning the volume of MARKET_BUY made on the candle. If high, the price tends to go up on
    # the next candle because it removes liquidity from the order book
    TAKER_BUY_VOLUME = 9
    # Taker buy volume on the current candle (ex : TAKER BUY VOLUME in USDT if BTCUSDT selected)
    # meaning the volume of MARKET_BUY made on the candle. If high, the price tends to go up on
    # the next candle because it removes liquidity from the order book
    TAKER_BUY_ASSET_VOLUME = 10

    def __init__(self):
        self.__init__()
