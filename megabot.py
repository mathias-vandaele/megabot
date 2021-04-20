"""
author : Mathias Vandaele
"""
import binance_f
import sys

from utils.data_management import data_management


def main():
    if sys.argv[1] == "train":
        print("training mode starting")
    elif sys.argv[1] == "run":
        print("running mode starting")
    elif sys.argv[1] == "backtest":
        print("back testing mode starting")
    elif sys.argv[1] == "freshdata":
        print("retrieving fresh data")
        data_management.retrieve_and_save_data("ETHUSDT", "1 Jan, 2020")


# Press the green button in the gutter to run the script
if __name__ == '__main__':
    main()