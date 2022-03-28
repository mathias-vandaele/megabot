"""
author : Mathias Vandaele
"""

import argparse
import sys
import tensorflow as tf

from neural_intelligence.smart_lstm import SmartLSTM
from utils.data_management import DataManagement
from neural_intelligence.lstm_network_multivariate_stateless import LstmNetworkMultivariateStateless


def main(parsed_args):
    if parsed_args.mode == "train":
        if parsed_args.pair is None:
            sys.exit("For this mode; you need to set --pair to select the pair you want to fetch "
                     "the data")
        print("training mode starting")
        """
        LSTM MULTIVARIATE STATELESS 
        one example of it : https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
        """
        if parsed_args.algo == "lstmstateless":
            lstm = LstmNetworkMultivariateStateless(parsed_args.pair, features=3, sequence_length=60, n_future=60)
            lstm.train()
            lstm.save()
            lstm.plot_forecast_vs_truth(parsed_args.pair)

        if parsed_args.algo == "smartlstm":
            smart_lstm = SmartLSTM(parsed_args.pair, sequence_length=30)
            #smart_lstm.train()
            #smart_lstm.save()
            smart_lstm.forecast_and_compare()

    elif parsed_args.mode == "run":
        print("running mode starting")
    elif parsed_args.mode == "backtest":
        print("back testing mode starting")
    elif parsed_args.mode == "freshdata":
        if parsed_args.pair is None:
            sys.exit("For this mode; you need to set --pair to select the pair you want to fetch "
                     "the data")
        if parsed_args.date is None:
            sys.exit("For this mode; you need to set --pair to select the date you want to fetch "
                     "the data from ")
        print("retrieving fresh data ...")
        DataManagement.retrieve_and_save_data(parsed_args.pair, parsed_args.date)


# Press the green button in the gutter to run the script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="The mode you want to use the bot (train/run/backtest/freshdata)")
    parser.add_argument("--algo", help="select the algorithm you want to use for the selected mode")
    parser.add_argument("--pair", help="The pair you want to run the bot on or retrieve datas")
    parser.add_argument("--date", help="If retrieving datas, set the date you want to get the datas from (ex : '1 Jan, 2020'")
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        tf.device("/cpu:0")
    else:
        print("Please install GPU version of TF")
    parsed_args = parser.parse_args()
    main(parsed_args)
