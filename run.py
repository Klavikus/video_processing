import argparse
import tomli
from video_processing import VideoProcessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")

    args = parser.parse_args()

    with open(args.config, mode="rb") as fp:
        config = tomli.load(fp)

    vp = VideoProcessing(config)
    vp.process()
