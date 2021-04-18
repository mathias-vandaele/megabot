import sys


def main():
    if sys.argv[1] == "train":
        print("training mode starting")
    elif sys.argv[1] == "run":
        print("running mode starting")
    elif sys.argv[1] == "backtest":
        print("back testing mode starting")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()