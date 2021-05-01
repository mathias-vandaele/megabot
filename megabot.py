"""
author : Mathias Vandaele
"""
import argparse
import sys

from tensorflow.python.estimator import inputs

from neural_intelligence.batches_generator import generate_batch
from utils.data_management import DataManagement
from neural_intelligence.lstm_network_multivariate_stateless import LstmNetworkMultivariateStateless


def main(parsed_args):
    if parsed_args.mode == "train":
        if parsed_args.pair is None:
            sys.exit("For this mode; you need to set --pair to select the pair you want to fetch "
                     "the data")
        print("training mode starting")
        inputs, targets = generate_batch("BTCUSDT")
        lstm = LstmNetworkMultivariateStateless(features=5)
        lstm.train(inputs, targets)
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
    parser.add_argument("--pair", help="The pair you want to run the bot on or retrieve datas")
    parser.add_argument("--date", help="If retrieving datas, set the date you want to get the datas from (ex : '1 "
                                       "Jan, 2020'")
    parsed_args = parser.parse_args()
    main(parsed_args)
