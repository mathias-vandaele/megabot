"""
author : Mathias Vandaele
"""
import argparse
import sys

from utils.data_management import data_management


def main(args):
    if args.mode == "train":
        print("training mode starting")
    elif args.mode == "run":
        print("running mode starting")
    elif args.mode == "backtest":
        print("back testing mode starting")
    elif args.mode == "freshdata":
        if args.pair is None:
            sys.exit("For this mode; you need to set --pair to select the pair you want to fetch the data")
        if args.date is None:
            sys.exit("For this mode; you need to set --pair to select the date you want to fetch the data from ")
        print("retrieving fresh data ...")
        data_management.retrieve_and_save_data(args.pair, args.date)


# Press the green button in the gutter to run the script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="The mode you want to use the bot (train/run/backtest/freshdata)")
    parser.add_argument("--pair", help="The pair you want to run the bot on or retrieve datas")
    parser.add_argument("--date", help="If retrieving datas, set the date you want to get the datas from (ex : '1 "
                                       "Jan, 2020'")
    args = parser.parse_args()
    main(args)
