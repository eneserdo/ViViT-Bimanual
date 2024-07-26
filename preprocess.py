# Run this file once when you first download the dataset to preprocess the data

import os
import glob
import argparse
import shutil
from dataset import STRIDE, crawl_dataset_FULL as crawl_dataset
from PIL import Image

def arg_parser():
    parser = argparse.ArgumentParser(description="Preprocess the dataset")
    parser.add_argument("-p", "--data_path", type=str, help="Path to the dataset")
    parser.add_argument("-rmd","--rm_depth", action="store_true", help="Remove depth images")
    parser.add_argument("-c","--crop", action="store_true", help="Crop the images")
    parser.add_argument("-ds", "--downsample", action="store_true", help="Downsample the images in temporal domain (2)")

    
    return parser.parse_args()


def preprocess_dataset(args):
    print("Remove depth: ", args.rm_depth, " Crop: ", args.crop, " Downsample: ", args.downsample)

    # copy the dataset to a new folder
    # print("Copying the dataset to a new folder...")
    # shutil.copytree(args.data_path, args.data_path+"_preprocessed")

    print("Preprocessing the dataset...")
    dataset_dir = args.data_path #+"_preprocessed"
    
    # delete depth images
    if args.rm_depth:
        for subject, task, take in crawl_dataset():
            depth_dir = os.path.join(dataset_dir, subject, task, take, "depth")
            shutil.rmtree(depth_dir)

    # downsample and crop the rgb images
    for subject, task, take in crawl_dataset():
        rgb_dir = os.path.join(args.data_path, subject, task, take, "rgb")
        frame_list = glob.glob(os.path.join(rgb_dir, "chunk_*", "*.png"))

        frame_list_selected = frame_list[::STRIDE]
        frame_list_deleted = list(set(frame_list) - set(frame_list_selected))
        
        if args.downsample:
            for frame in frame_list_deleted:
                os.remove(frame)

        if args.crop:
            for frame in frame_list_selected:
                img = Image.open(frame)
                img = img.crop(...)
                img.save(frame)

    print("OK!")
    
def main():
    args = arg_parser()
    preprocess_dataset(args)


if __name__ == "__main__":
    main()