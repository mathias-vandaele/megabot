"""
author : Mathias Vandaele
"""
import os

from binance.client import Client


class constants:

    api_key = os.environ.get('binance_api')
    api_secret = os.environ.get('binance_secret')
    kline_interval = Client.KLINE_INTERVAL_1MINUTE

    CANDLE_TIMESTAMP = 0
    CANDLE_OPEN = 1
    CANDLE_HIGH = 2
    CANDLE_LOW = 3
    CANDLE_CLOSE = 4

    def __init__(self):
        self.__init__()
