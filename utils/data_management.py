"""
author : Mathias Vandaele
"""
import numpy as np

from api.binance_data import binance_data
from constants.constants import constants


class data_management:

    def __init__(self):
        self.__init__()

    @staticmethod
    def retrieve_and_save_data(pair, since):
        ibinance_datas = binance_data()
        datas = ibinance_datas.retrieve_data(pair, constants.kline_interval, since)
        np.savetxt("resources/" + pair + ".csv", datas, delimiter=',', fmt='%s')
