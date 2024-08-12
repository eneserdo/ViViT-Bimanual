from functools import partial
import glob
import json
import os
import pickle
from os.path import join as opj
from typing import Callable, List, Tuple

import natsort
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import transformers
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

# Global variables

DEBUG_MODE = False

STRIDE = 3
CLIP_LEN = 30
# ViViT will process CLIP_LEN//STRIDE frames at a time

if os.path.exists("/home/cak/enes/"):   
    DATASET_DIR = "/home/cak/enes/bimanual/bimacs_rgbd_data/ds2_bimacs_rgbd_data"
    GT_DIR = "/home/cak/enes/Online-Action-Prediction-from-Scene-Graphs/raw/bimacs_rgbd_data_ground_truth"
    print("[CAK]")
elif os.path.exists("/home/stannis/enes/"):
    DATASET_DIR = "/home/stannis/enes/bimanual/ds2_bimacs_rgbd_data"
    GT_DIR = "/home/stannis/Online-Action-Prediction-from-Scene-Graphs/raw/bimacs_rgbd_data_ground_truth"
    print("[STANNIS]")
elif os.path.exists("/YEREL/ds2_bimacs_rgbd_data"):
    DATASET_DIR = "/YEREL/ds2_bimacs_rgbd_data"
    GT_DIR = "/YEREL/bimacs_rgbd_data_ground_truth"
    print("[UHEM]")
elif os.path.exists("/ari/users/eerdogan03/ds2_bimacs_rgbd_data"):
    DATASET_DIR = "/ari/users/eerdogan03/ds2_bimacs_rgbd_data"
    GT_DIR = "/ari/users/eerdogan03/bimacs_rgbd_data_ground_truth"
    print("[ARI]")

if DEBUG_MODE:
    DATASET_DIR+="_debug"

assert os.path.exists(DATASET_DIR), f"{DATASET_DIR} not found"
assert os.path.exists(GT_DIR), f"{GT_DIR} not found"

OBJECTS = [
    "bowl",
    "knife",
    "screwdriver",
    "cuttingboard",
    "whisk",
    "hammer",
    "bottle",
    "cup",
    "banana",
    "cereals",
    "sponge",
    "woodenwedge",
    "saw",
    "harddrive",
    "LeftHand",
    "RightHand",
]
RELATIONS = [
    "contact",
    "above",
    "below",
    "left of",
    "right of",
    "behind of",
    "in front of",
    "inside",
    "surround",
    "moving together",
    "halting together",
    "fixed moving together",
    "getting close",
    "moving apart",
    "stable",
]  # , 'temporal'

ACTIONS = [
    "idle",
    "approach",
    "retreat",
    "lift",
    "place",
    "hold",
    "pour",
    "cut",
    "hammer",
    "saw",
    "stir",
    "screw",
    "drink",
    "wipe",
]

TASKS = [
    "task_1_k_cooking",
    "task_2_k_cooking_with_bowls",
    "task_3_k_pouring",
    "task_4_k_wiping",
    "task_5_k_cereals",
    "task_6_w_hard_drive",
    "task_7_w_free_hard_drive",
    "task_8_w_hammering",
    "task_9_w_sawing",
]




def crawl_dataset_FULL():
    for subject in ["subject_{}".format(i) for i in range(1, 7)]:
        for task in TASKS:
            for take in ["take_{}".format(take_i) for take_i in range(10)]:
                yield subject, task, take


def crawl_dataset_DEBUG():
    for subject in ["subject_1", "subject_2"]:  # (1, 7)
        for task in ['task_1_k_cooking', 'task_2_k_cooking_with_bowls']:
            for take in ["take_1", "take_7"]:  # 10
                yield subject, task, take




if DEBUG_MODE:
    print("Debug mode is on\n" * 5)
    crawl_dataset = crawl_dataset_DEBUG
else:
    crawl_dataset = crawl_dataset_FULL


def _get_actions_list(gt_action_json: os.PathLike, act_len: int) -> np.ndarray:
    # Returns the action of left hand and right hand, respectively.

    with open(gt_action_json) as f:
        actions_json = json.load(f)

    actions_list = np.zeros((act_len, 2), dtype=np.int32) # FIXME: dont use act_len

    # right
    for t in range(0, len(actions_json["right_hand"]) - 1, 2):

        begin = actions_json["right_hand"][t]
        right_act = actions_json["right_hand"][t + 1]
        end = actions_json["right_hand"][t + 2]

        right_act = -1 if right_act is None else right_act

        actions_list[begin:end, 1] = right_act

    # left
    for t in range(0, len(actions_json["left_hand"]) - 1, 2):

        begin = actions_json["left_hand"][t]
        left_act = actions_json["left_hand"][t + 1]
        end = actions_json["left_hand"][t + 2]

        left_act = -1 if left_act is None else left_act
        actions_list[begin:end, 0] = left_act

    return actions_list


class BimanualDataset(Dataset):
    def __init__(self, args, sample_dict, gt_dict, split: str):

        if args.model == "b16":
            img_size = 224
        elif args.model == "b16se":
            img_size = 384
        elif args.model == "h14se":
            img_size = 518
        elif args.model == "l16":
            img_size = 224
        elif args.model == "l16se":
            img_size = 512


        transforms = [torchvision.transforms.CenterCrop(480),
                      torchvision.transforms.Resize(img_size, antialias=True)]
        
        if args.use_randaug and split == "train": 
            transforms.append(torchvision.transforms.RandAugment())            

        self.transform_wo_norm = torchvision.transforms.Compose(transforms + [torchvision.transforms.Lambda(lambda x: x.permute(0, 2, 3, 1))])
        
        transforms.extend([torchvision.transforms.ConvertImageDtype(torch.float), # [0, 255] -> [0.0, 1.0]
                           torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225]) ])
        
        self.transform = torchvision.transforms.Compose(transforms)

        self.split = split
        # self.test_subject = args.test_subject
        # self.validation_take = args.validation_take
        self.sample_dict = sample_dict
        self.gt_dict = gt_dict
        self.split_len = len(sample_dict)


    def __len__(self):
        return self.split_len


    def __getitem__(self, idx):

        subject, task, take, frame_idx, _ = self.sample_dict[idx]
        
        act_list = self.gt_dict[str(subject) + str(task) + str(take)]

        labels = act_list[frame_idx:frame_idx+CLIP_LEN*STRIDE]
        labels = labels[::STRIDE]


        imgs = []
        for f in range(frame_idx, frame_idx + CLIP_LEN*STRIDE, STRIDE): # 0, 2, 4 ... 62
            path = f"{DATASET_DIR}/subject_{subject}/{TASKS[task-1]}/take_{take}/rgb/chunk_{f//100}/frame_{f}.png"
            
            imgs.append(F.pil_to_tensor(Image.open(path)))
        

        assert len(imgs) == CLIP_LEN == len(labels), f"{len(imgs)}, {CLIP_LEN}, {len(labels)}"

        imgs = torch.stack(imgs)
        imgs = self.transform(imgs)

        labels = torch.tensor(labels, dtype=torch.long)

        return imgs, labels, idx

    def get_original_imgs(self, idx) -> Tuple[List[np.ndarray], torch.Tensor]:
        # This function is identical to __getitem__ but it returns the original images

        subject, task, take, frame_idx, _ = self.sample_dict[idx]


        imgs = []
        for f in range(frame_idx, frame_idx + CLIP_LEN*STRIDE, STRIDE): # 0, 2, 4 ... 62
            path = f"{DATASET_DIR}/subject_{subject}/{TASKS[task-1]}/take_{take}/rgb/chunk_{f//100}/frame_{f}.png"

            imgs.append(F.pil_to_tensor(Image.open(path)))            

        
        imgs = torch.stack(imgs)
        imgs = self.transform_wo_norm(imgs)

        return imgs

def make_mappings(args):
    # saves a dict for sample_idx -> img path, img idx
    # saves a dict for vid_idx -> action list


    sample_dict_master = {}  # sample_idx -> (subject, task, take, frame_idx)
    gt_dict = {}

    sample_idx = 0

    for subject, task, take in crawl_dataset():
        subject_int, task_int, take_int = int(subject[-1]), int(task[5]), int(take[-1])

        rgb_dir = opj(DATASET_DIR, subject, task, take, "rgb")

        frame_list = natsort.natsorted(glob.glob(opj(rgb_dir, "chunk_*", "*.png")))

        act_len = int(frame_list[-1].split("_")[-1][:-4]) + 1

        act_list = _get_actions_list(
            opj(GT_DIR, subject, task, f"{take}.json"), act_len
        )

        # assert len(act_list) == len(frame_list)

        gt_dict[str(subject_int) + str(task_int) + str(take_int)] = act_list # must be full fps

        # frame_list = frame_list[::STRIDE] 
        # No need to downsample here because we did this in preprocess.py
        
        frame_idx = 0

        for frame in frame_list[: -CLIP_LEN + 1]:
            assert frame.split("_")[-1][:-4] == str(frame_idx), f"{frame.split('_')[-1][:-4]} != {frame_idx}, vid: {subject_int} {task_int} {take_int}"

            sample_dict_master[sample_idx] = (
                subject_int,
                task_int,
                take_int,
                frame_idx,
                act_list[frame_idx]
            )
            sample_idx += 1
            frame_idx += STRIDE

    with open("data/sample_dict_master.pickle", "wb") as f:
        pickle.dump(sample_dict_master, f)

    with open("data/gt_dict.pickle", "wb") as f:
        pickle.dump(gt_dict, f)


def make_fold_mappings(test_subject: int, validation_take: int):
    # read master mappings
    with open("data/sample_dict_master.pickle", "rb") as f:
        sample_dict_master = pickle.load(f)

    sample_dict = {"train": {}, "test": {}, "val": {}}
    test_idx = 0
    val_idx = 0
    train_idx = 0

    for idx, (subject, task, take, frame_idx, actions) in sample_dict_master.items():
        if subject == test_subject:
            sample_dict["test"][test_idx] = (subject, task, take, frame_idx, actions)
            test_idx += 1

        elif take == validation_take:
            sample_dict["val"][val_idx] = (subject, task, take, frame_idx, actions)
            val_idx += 1

        else:
            sample_dict["train"][train_idx] = (subject, task, take, frame_idx, actions)
            train_idx += 1

    assert len(sample_dict["test"]) + len(sample_dict["val"]) + len(sample_dict["train"]) == len(sample_dict_master), f"{len(sample_dict['test'])} + {len(sample_dict['val'])} + {len(sample_dict['train'])} != {len(sample_dict_master)}"

    with open(f"data/sample_dict_fold{test_subject}.pickle", "wb") as f:
        pickle.dump(sample_dict, f)


def get_bimanual_dataset_splits(args) -> Tuple[Dataset, Dataset, Dataset]:
    # Create mappings, but only use relevant fold mappings

    files_to_check = ["data/sample_dict_master.pickle", "data/gt_dict.pickle"]
    files_to_check += [f"data/sample_dict_fold{i}.pickle" for i in range(1, 7)]

    if not all([os.path.exists(f) for f in files_to_check]):
        print("Creating mappings...")

        make_mappings(args)
        for i in range(1, 7):
            make_fold_mappings(i, args.validation_take)
    else:
        print("Using existing mappings...")

    with open(f"data/sample_dict_fold{args.test_subject}.pickle", "rb") as f:
        sample_dict_fold = pickle.load(f)
    
    with open("data/gt_dict.pickle", "rb") as f:
        gt_dict = pickle.load(f)

    train_set = BimanualDataset(args, sample_dict_fold["train"], gt_dict, "train")
    test_set = BimanualDataset(args, sample_dict_fold["test"], gt_dict, "test")
    val_set = BimanualDataset(args, sample_dict_fold["val"], gt_dict, "val")

    return train_set, test_set, val_set
