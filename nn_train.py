import argparse
import tomli
from nn_denoise import NNDenoise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("-c", "--config", action="store_true")

    args = parser.parse_args()

    with open(args.config, mode="rb") as fp:
        config = tomli.load(fp)

    nn = NNDenoise(config)
    nn.train()
