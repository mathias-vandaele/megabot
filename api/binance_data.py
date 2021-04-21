import numpy as np
from binance.client import Client
import constants.constants


class BinanceData:

    def __init__(self):
        self.client = Client(constants.constants.Constants.api_key, constants.constants.Constants.api_secret)

    def retrieve_data(self, pair, kline_interval, since):
        kline = np.array(self.client.get_historical_klines(pair, kline_interval, since))
        return kline
