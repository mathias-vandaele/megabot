import numpy as np
from binance.client import Client
from constants.constants import Constants


class BinanceData:

    def __init__(self):
        self.client = Client(Constants.api_key, Constants.api_secret)

    def retrieve_data(self, pair, kline_interval, since):
        kline = np.array(self.client.get_historical_klines(pair, kline_interval, since))
        return kline
