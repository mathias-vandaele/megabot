"""
author : Mathias Vandaele
"""
import math

import numpy as np


class RSI:
    def __init__(self):
        self.__init__()

    @staticmethod
    def calculate_rsi(data, lookback):
        RSI = np.zeros((len(data), 1))
        for i in range(len(data)):
            if i - lookback < 0:
                continue
            else:
                differences_up = []
                differences_down = []
                for j in reversed(range(lookback)):
                    # print("n-1 : ", str(data[i - j - 1][0]))
                    # print("n : ", str(data[i - j][0]))
                    percentage_increase = ((data[i - j][0] - data[i - j - 1][0]) / data[i - j - 1][0]) * 100
                    # print("percentage_increase ", percentage_increase)
                    if percentage_increase >= 0:
                        differences_up.append(percentage_increase)
                    else:
                        differences_down.append(abs(percentage_increase))

                mean_up = 0 if math.isnan(np.mean(differences_up)) else np.mean(differences_up)
                mean_down = 0 if math.isnan(np.mean(differences_down)) else np.mean(differences_down)

                current_rsi = 100 - ((100 / (1 + ((mean_up / lookback) / (mean_down / lookback)))))
                RSI[i][0] = current_rsi

        data = np.hstack((data, RSI))
        return data
