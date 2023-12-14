import sys
from train_model import train_model
from find_anomalies import find_anomalies


if __name__ == "__main__":
    train_input = sys.argv[1]
    test_input = sys.argv[2]

    train_model(train_input)
    find_anomalies(train_input, test_input)
