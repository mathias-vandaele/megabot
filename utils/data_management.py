"""
author : Mathias Vandaele
"""
import numpy as np

from api.binance_data import BinanceData
from constants.constants import Constants


class DataManagement:

    def __init__(self):
        self.__init__()

    @staticmethod
    def retrieve_and_save_data(pair, since):
        instance_binance_data = BinanceData()
        data = instance_binance_data.retrieve_data(pair, Constants.kline_interval, since)
        np.savetxt("resources/" + pair + ".csv", data, delimiter=',', fmt='%s')
