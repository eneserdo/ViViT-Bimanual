# Run this file once when you first download the dataset to preprocess the data
import os
import glob
import argparse
from pathlib import Path
import shutil
from time import sleep

from tqdm import tqdm
from dataset import STRIDE, crawl_dataset_FULL as crawl_dataset
from PIL import Image

from natsort import natsorted

def arg_parser():
    parser = argparse.ArgumentParser(description="Preprocess the dataset")
    parser.add_argument("-p", "--data_path", type=str, help="Path to the dataset")
    parser.add_argument("-rmd","--rm_depth", action="store_true", help="Remove depth images")
    parser.add_argument("-c","--crop", action="store_true", help="Crop the images")
    parser.add_argument("-ds", "--downsample", type=int, default=1, help="Downsample the images in temporal axis (2)")

    
    return parser.parse_args()


def preprocess_dataset(args):

    if STRIDE != args.downsample:
        print("Warning: STRIDE is different from the downsample rate", STRIDE, args.downsample)
        sleep(3)

    print("Remove depth:", args.rm_depth, "Crop:", args.crop, "Downsample:", args.downsample)

    args.data_path = Path(args.data_path)
    dataset_dir = args.data_path.parent / (f"ds{args.downsample}_" + args.data_path.name)

    # copy the dataset to a new folder
    print("Copying the dataset to a new folder...")
    shutil.copytree(args.data_path, dataset_dir)
    
    # delete depth images
    if args.rm_depth:
        for subject, task, take in crawl_dataset():
            depth_dir = os.path.join(dataset_dir, subject, task, take, "depth")
            shutil.rmtree(depth_dir)

    # downsample and crop the rgb images
    for subject, task, take in tqdm(crawl_dataset(), desc="Preprocessing", total=540):
        rgb_dir = os.path.join(dataset_dir, subject, task, take, "rgb")
        frame_list = natsorted(glob.glob(os.path.join(rgb_dir, "chunk_*", "*.png")))

        frame_list_selected = frame_list[::STRIDE]
        frame_list_deleted = list(set(frame_list) - set(frame_list_selected))
        

        for frame in frame_list_selected:
            assert int(frame.split("/")[-1][6:-4]) % STRIDE == 0, "Frame number is not divisible by STRIDE"

            need_to_save = False
            
            img = Image.open(frame)
            if args.crop:
                img = img.crop((80, 0, 80+480, 480))
                need_to_save = True

            if img.mode == "RGBA":
                img = img.convert("RGB")
                need_to_save = True
                
            elif img.mode == "RGB":
                pass
            else:
                raise ValueError("Unknown image mode: ", img.mode)
            
            if need_to_save:
                img.save(frame)

        if args.downsample:
            for frame in frame_list_deleted:
                os.remove(frame)
    
    print("OK!")
    
def main():
    args = arg_parser()
    preprocess_dataset(args)


if __name__ == "__main__":
    main()