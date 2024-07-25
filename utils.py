import argparse
import glob
import json
from math import ceil
import os
import random
import shutil
import warnings
from os.path import join as opj

import matplotlib.pyplot as plt
import natsort
import numpy as np
import seaborn as sns
import torch
from torchmetrics.classification import (MulticlassConfusionMatrix,
                                         MulticlassF1Score)


from scipy.optimize import linear_sum_assignment

from dataset import ACTIONS, STRIDE
# from metrics import f1_at_k_single_example
# from visualization_utils import (plot_segmentation, plot_segmentation_partial,
#                                  plot_segmentation_v2)

plt.ioff()


def arg_parser_base():
    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("-ts", "--test_subject", type=int, default=1, choices=[1, 2, 3, 4, 5, 6], help="...")
    ap.add_argument("-vt", "--validation_take", type=int, default=7, choices=[i for i in range(10)], help="...")

    # Not implemented yet, for torch it is default
    # ap.add_argument("--use_randaug", action="store_true", help="Use RandAugment for data augmentation")

    # Model
    ap.add_argument("-bs", "--batch_size", type=int, default=1, help="...")
    ap.add_argument("--model", type=str, default="hf", choices=["hf", "tr"], help="...")

    # if model == "hf"
    # ap.add_argument("-c", "--model-config", type=str, default="google/vivit-b-16x2-kinetics400")

    # if model == "tr"
    ap.add_argument("--nhead", type=int, default=12, help="For temporal encoder")
    ap.add_argument("--dim_feedforward", type=int, default=3072, help="For temporal encoder")
    ap.add_argument("--num_layers", type=int, default=12, help="For temporal encoder")
    
    # Optimizer
    ap.add_argument("-o", "--optimizer_type", type=str,
                    choices=["Adam", "AdamW", "NAdam", "RMSprop", "SGD"],
                    help="Type of optimizer to be used  from torch.optim", default='SGD')
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="...")
    ap.add_argument("-mn", "--momentum", type=float, default=0.9, help="...")
    ap.add_argument("-wd", "--weight_decay", type=float, default=0.0001, help="...")
    ap.add_argument("--scheduler_step_size", type=int, default=-1, help="...")
    ap.add_argument("--scheduler_gamma", type=float, default=0.5, help="...")
    ap.add_argument("--weighted_loss", type=str, default="none", choices=["none", "inv", "sqrt_inv", "effective"],
                    help="Use class frequencies as weight for loss function")       # :(
    ap.add_argument("--smoothing", type=float, default=0.0, help="Label smoothing weight")
    ap.add_argument("--num_acc", type=int, default=1, help="Number of accumulations before backward")
    ap.add_argument("--patience", type=int, default=2, help="patience for ReduceLROnPlateau scheduler")
    ap.add_argument("--clip", type=float, default=-1, help="Gradient clipping")
    ap.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")


    # Metrics
    ap.add_argument("--weighted_mv", type=str, default="none", choices=["none", "w1", "w2", "auto"], help="...")

    # Training
    ap.add_argument("--num_epochs", type=int, default=20, help="Max number of epochs")

    # Logging
    ap.add_argument("-n", "--name", required=True, type=str, help="Name of experiment for logging into Wandb")
    ap.add_argument("--disable_wandb",action="store_true", help="Disable wandb")
    ap.add_argument("--save_incorrects",action="store_true", help="saves some incorrect samples")
    ap.add_argument("--save_individual",action="store_true", help="Visualize individual segment predictions")
    ap.add_argument("--save_segments",action="store_true", help="Visualize final predictions for whole video")
    ap.add_argument("--saving_freq", type=int, default=2, help="...")

    # Others
    ap.add_argument("--gpu_id", type=int, default=0, help="...")
    ap.add_argument("--seed", type=int, default=34, help="...")

    return ap

def arg_parser():
    ap = arg_parser_base()
    return ap.parse_args()

def arg_parser_cv():
    ap = arg_parser_base()

    # Cross validation 
    ap.add_argument("--num_folds", type=int, default=6, help="Number of folds")
    
    return ap.parse_args()


def get_minutes(elapsed) -> str:
    return f"{round((elapsed) // 60)}.{round((elapsed) % 60)}"


def plot_conf_mat(mat, name, do_save=False, do_return_plot=False):
    if isinstance(mat, torch.Tensor):
        mat = mat.cpu().numpy()
    
    mat_long = np.round(mat * 100).astype(int)

    plt.figure(figsize=(14, 12))
    sns.heatmap(mat_long, square=True, annot=True, cmap='Blues', cbar=False, xticklabels=ACTIONS, yticklabels=ACTIONS,
                vmin=0, vmax=100)
    plt.yticks(rotation=0)
    plt.title(f"{name}")
    plt.xlabel('Predicted')
    plt.ylabel('Ground truth')
    if do_save:
        raise NotImplementedError("conf_mat diye bir klasor yok artik")
        plt.savefig(opj("conf_mat", f'{name}.png'))
    if do_return_plot:
        return plt
    else:
        plt.close()

def plot_line(line, name, do_save=False, do_return_plot=False):
    
    # assert len(line.shape)==1, "Line must be 1D"

    line = np.array(line)*100

    plt.figure(figsize=(14, 12))
    plt.plot(line)
    plt.yticks(np.arange(min(line)-1, max(line)+1, 1.0))
    plt.title(name)
    plt.axhline(y = np.max(line), linestyle = '--') 
    plt.xlabel('t')
    plt.ylabel('F1')
    if do_save:
        raise NotImplementedError("conf_mat diye bir klasor yok artik")
        plt.savefig(opj("conf_mat", f'{name}.png'))
    if do_return_plot:
        return plt
    else:
        plt.close()


def create_ground_truth_segments(label_series):
    change_occurring = torch.concat([label_series[1:] - label_series[:-1], torch.zeros(size=(1, label_series.shape[-1]), device=label_series.device)], dim=0)
    return torch.where(change_occurring != 0, 1.0, 0.0) # Should this be of shape [N, B, 2] instead of [N, B*2]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_filters_for_imbalance(dataBatchList):
    """ 2 out of 3 samples for the actions idle and hold were discarded """

    batch_size = dataBatchList[0].y.shape[0]
    left_filter = torch.full((len(dataBatchList), batch_size // 2), True)  # shape = [sequence_length, batch_size]
    right_filter = torch.full((len(dataBatchList), batch_size // 2), True)  # shape = [sequence_length, batch_size]

    for i in range(batch_size // 2):
        count_idle_left = 0
        count_idle_right = 0
        count_hold_left = 0
        count_hold_right = 0

        for t, dataBatch in enumerate(dataBatchList):
            count_idle_left = (count_idle_left + torch.argmax(dataBatch[i].y[0]) == ACTIONS.index("idle")) % 3
            count_idle_right = (count_idle_right + torch.argmax(dataBatch[i].y[1]) == ACTIONS.index("idle")) % 3
            count_hold_left = (count_hold_left + torch.argmax(dataBatch[i].y[0]) == ACTIONS.index("hold")) % 3
            count_hold_right = (count_hold_right + torch.argmax(dataBatch[i].y[1]) == ACTIONS.index("hold")) % 3

            left_filter[t, i] = not (count_idle_left in [1, 2])
            right_filter[t, i] = not (count_idle_right in [1, 2])

            left_filter[t, i] = not (count_hold_left in [1, 2])
            right_filter[t, i] = not (count_hold_right in [1, 2])

    return left_filter, right_filter


def align_embeddings_to_right(x, mask):
    """
        Ref 1: https://discuss.pytorch.org/t/reorder-non-zero-values-in-a-2d-matrix/112314
        Ref 2: https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    """

    emd_size = x.shape[2]
    mask = mask[:, :, None]

    # Somehow when it is on gpu it gives wrong values, to fix it I used "stable=True"
    # Still suspicious
    index = torch.sort(~mask.repeat(1, 1, emd_size).to(x.device) * 1, dim=0, descending=True, stable=True)[1]
    y = x.gather(0, index)

    # d_mask = torch.sum(mask, dim=0, keepdims=True) > torch.arange(x.shape[0]).reshape(-1,1)
    # y = torch.zeros_like(x)
    # y[d_mask] = x[mask]

    return y


def debug_fih(dataBatchList, b):
    for i in range(10):

        # hold
        if torch.argmax(dataBatchList[i].y[b]).item() == 5:
            print("h ", end="")
        # idle
        elif torch.argmax(dataBatchList[i].y[b]).item() == 0:
            print("i ", end="")
        else:
            print(torch.argmax(dataBatchList[i].y[b]).item(), end=" ")

    print("")


def print_non_uniform_ys(dataBatchList):
    bs = dataBatchList[0].y.shape[0]
    for b in range(bs):
        act = torch.argmax(dataBatchList[0].y[b]).item()
        for i in range(10):
            if act != torch.argmax(dataBatchList[i].y[b]).item():
                print("b id: ", b)
                debug_fih(dataBatchList, b)
                break


def debug_align(device="cpu"):
    mask = torch.full((4, 3), True)
    mask[0, 0] = False
    mask[2, 2] = False
    mask[3, 0] = False
    mask[3, 1] = False

    # print("mask:\n", mask)

    x = torch.arange(12).reshape(4, 3, 1)
    x = x.repeat(1, 1, 2)
    x[~mask] = 0

    x = x.to(device)
    mask = mask.to(device)

    # t=4, bs=3, emb_size=2

    # print("x:\n", x)
    # print("x[:, :, 0]:\n", x[:, :, 0])

    # y = align_embeddings_to_right(x, mask)
    emd_size = x.shape[2]
    mask = mask[:, :, None]

    # Somehow when it is on gpu it gives wrong values
    index = torch.sort(~mask.repeat(1, 1, emd_size).to(x.device) * 1, dim=0, descending=True)[1]
    y = x.gather(0, index)

    # print("y[:, :, 0]:\n", y[:, :, 0])

    return x, y, index, mask


def get_relations(base_dir, subject, task, take, i):
    """ Returns a set for relations in the frame """

    # derived_dir=base_dir+"bimacs_derived_data/"
    derived_dir = opj(base_dir, "bimacs_derived_data")

    _3d_objects_dirs = opj(derived_dir, subject, task, take, "3d_objects")
    _3d_objects_json_list = natsort.natsorted(glob.glob(opj(_3d_objects_dirs, "*.json")))

    relation_json_dirs = opj(derived_dir, subject, task, take, "spatial_relations")
    relations_path = natsort.natsorted(glob.glob(opj(relation_json_dirs, "*.json")))

    with open(relations_path[i]) as f:
        relations = json.load(f)

    with open(_3d_objects_json_list[i]) as f:
        _3d = json.load(f)

    # log=""
    # text=""
    readable_relations = []

    for relation in relations:
        oi = relation["object_index"]
        si = relation["subject_index"]

        # log = log + _3d[oi]["class_name"]+ " - " + relation["relation_name"] + " - " + _3d[si]["class_name"] +"; "
        # log = log + _3d[oi]["class_name"]+ " - " + relation["relation_name"] + " - " + _3d[si]["class_name"] +"; "

        readable_relations.append(
            _3d[oi]["class_name"] + " - " + relation["relation_name"] + " - " + _3d[si]["class_name"])

        # if relation["relation_name"]=="contact":

        # text=text + f"({oi})"+object_mapping[oi] + " - " + f"({si})"+ object_mapping[si] + "\n"
        # if oi < si:
        #     text=text + _3d[oi]["class_name"]+ " - " + _3d[si]["class_name"] +"\n"

    # print("FRAME:\n", log, "\n")
    return readable_relations


def visualize_incorrect_preds(preds, labels, idxes, test_idx_sample_mapping, graph_dataset_dir="raw",
                              rgb_dataset_dir="/home/stannis/enes/bimanual/bimacs_rgbd_data"):
    assert os.path.exists(rgb_dataset_dir), "RGB dataset directory does not exist"
    assert os.path.exists(graph_dataset_dir), "Graph dataset directory does not exist"

    save_dir_name = "incorrect_predictions"
    # if os.path.exists(save_dir_name):
    # print("Removing existing", save_dir_name)
    # shutil.rmtree(save_dir_name)
    os.makedirs(save_dir_name, exist_ok=True)

    incorrect_mask = preds != labels
    preds, labels, idxes = preds[incorrect_mask], labels[incorrect_mask], idxes[incorrect_mask]

    num_of_samples = 10
    random_samples = np.random.randint(0, len(preds), num_of_samples)
    preds, labels, idxes = preds[random_samples], labels[random_samples], idxes[random_samples]

    for idx, p, gt in zip(idxes, preds, labels):
        subject, task, take, start_frame, end_frame, is_mirrored = test_idx_sample_mapping[idx]
        sample_save_dir = opj(save_dir_name, f"{ACTIONS[p]}_{ACTIONS[gt]}_{idx}")
        os.mkdir(sample_save_dir)

        # Copy frames
        for frame in range(start_frame, end_frame + 1):
            chunk_no = frame // 100
            frame_path = opj(rgb_dataset_dir, subject, task, take, "rgb", f"chunk_{chunk_no}", f"frame_{frame}.png")
            shutil.copy(frame_path, sample_save_dir)

        # Open a text file, and save relations
        with open(opj(sample_save_dir, "relations.txt"), "w") as f:
            prev_relations = []
            for frame in range(start_frame, end_frame + 1):
                f.write(f"Frame {frame}:\n\n")

                curr_relations = get_relations(graph_dataset_dir, subject, task, take, frame)

                diff_relations = list(set(curr_relations) - set(prev_relations))

                for relation in diff_relations:
                    f.write(relation + "; ")
                f.write("\n")

                prev_relations = curr_relations

def calculate_boundaries(arr):
    """ Sparse array to dense boundaries """

    id_changes = torch.where(arr ==1)[0]
    boundaries = torch.zeros((len(id_changes), 2), dtype=torch.int32)

    boundaries[1:, 0] = id_changes[:-1]+1
    boundaries[:, 1] = id_changes

    return boundaries


def calc_cost_matrix_single(gt_seg, pred_seg):
    """ Calculates cost matrix using hungarian algorithm for single example """
    # old:
    # gt_seg_pseudo_label=np.cumsum(gt_seg)
    # pred_seg_pseudo_label=np.cumsum(pred_seg)

    # stacked=np.stack((gt_seg_pseudo_label, pred_seg_pseudo_label), axis=1)
    # stacked = stacked.tolist()
    # stacked= [tuple(s) for s in stacked]
    # cntr=Counter(stacked)
    # m=sparse.csc_matrix((list(cntr.values()), list(cntr.keys())))
    # return m.toarray()
    # new: 
    # TODO add assert for gt_seg and pred_seg shapes 
    ground_truth_boundaries = calculate_boundaries(gt_seg)
    predicted_boundaries = calculate_boundaries(pred_seg)
    
    cost_matrix = torch.zeros((len(ground_truth_boundaries), len(predicted_boundaries)))

    for i, (gt_start, gt_end) in enumerate(ground_truth_boundaries):
        for j, (p_start, p_end) in enumerate(predicted_boundaries):

            # it is already passed
            # g: [+---+      ]
            # p: [       +--+]
            if gt_end < p_start:
                break

            # it is not reached yet
            # g: [       +--+]
            # p: [+---+      ]
            elif p_end < gt_start:
                continue

            # g: [ +----------+  ]
            # p: [    +---+      ]
            elif gt_start <= p_start and p_end <= gt_end:
                # cost_matrix[i, j] = p_end - p_start + 1
                cost_matrix[i, j] = (p_end - p_start + 1) / (gt_end - gt_start + 1)
            
            # g: [    +---+      ]
            # p: [ +----------+  ]
            elif p_start <= gt_start and gt_end <= p_end:
                # cost_matrix[i, j] = gt_end - gt_start + 1
                cost_matrix[i, j] = (gt_end - gt_start + 1) / (p_end - p_start + 1)
            
            # g: [   +--------+   ]
            # p: [ +----+         ]
            # or
            # g: [ +----+         ]
            # p: [   +--------+   ] 
            else:
                arg1 = p_end - gt_start + 1
                arg2 = gt_end - p_start + 1
                assert arg1 > 0 and arg2 > 0, "arg1 and arg2 must be positive"
                # cost_matrix[i, j] = min(max(arg1, 0), max(arg2, 0)) 
                cost_temp = arg1/arg2 if arg1 < arg2 else arg2/arg1
    
    return cost_matrix, ground_truth_boundaries, predicted_boundaries
    

def assign_seg_labels_single(cost_matrix, gt_seg_labels, gt_boundaries, pred_boundaries):
    """ Assigns ground truth segment labels to predicted unknown segments for single example """

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    pred_seg_labels = torch.zeros_like(gt_seg_labels)

    for i, j in zip(row_ind, col_ind):
        label_loc, _ = gt_boundaries[i]
        label = gt_seg_labels[label_loc]
        start, end = pred_boundaries[j]
        pred_seg_labels[start:end] = label

    return pred_seg_labels


def assign_seg_labels(gt_seg_labels, gt_seg, pred_seg):
    """ Assigns ground truth segment labels to predicted unknown segments """
    # gt_seg_labels: [T, B, C]
    # gt_seg: [T, B] (ones and zeros, ones indicate segment endings)
    # pred_seg: [T, B] (ones and zeros, ones indicate segment endings)
    assert gt_seg_labels.shape[:2] == gt_seg.shape == pred_seg.shape, "Shapes must be equal"

    # calculate cost matrix
    gt_seg_labels = gt_seg_labels.argmax(axis=2)
    pred_seg = pred_seg > 0.5
    
    # FIXME: this is a hack to make sure that the last frame is always a segment ending
    gt_seg[-1, :] = 1
    pred_seg[-1, :] = 1
    
    batch_size=gt_seg.shape[1]
    pred_seg_labels = torch.zeros_like(gt_seg_labels)

    for b_id in range(batch_size):
        gt_seg_labels_b = gt_seg_labels[:, b_id]
        gt_seg_b = gt_seg[:, b_id]
        pred_seg_b = pred_seg[:, b_id]

        # calculate cost matrix
        cost_matrix, gt_boundaries, pred_boundaries = calc_cost_matrix_single(gt_seg_b, pred_seg_b)

        # apply hungarian algorithm
        # assign labels to unknown segments
        pred_seg_labels_b = assign_seg_labels_single(cost_matrix, gt_seg_labels_b, gt_boundaries, pred_boundaries)
        pred_seg_labels[:, b_id] = pred_seg_labels_b

    return pred_seg_labels


class F1_at_k:
    # Very custom class for calculating f1@k
    def __init__(self, num_classes, batch_mode=False):
        self.num_classes = num_classes
        self.k_10 = []
        self.k_25 = []
        self.k_50 = []
        
        # To prevent bugs
        self.batch_mode = batch_mode

        
    def __call__(self, y_true, y_pred, segment_name = None):
         
        assert y_true.shape == y_pred.shape, f"{y_true.shape} != {y_pred.shape}"
        
        # if self.batch_mode:
        #     for y_t, y_p in zip(y_true, y_pred):
        #         self.k_10.append(f1_at_k_single_example(y_t, y_p, num_classes=self.num_classes, overlap=0.1))
        #         self.k_25.append(f1_at_k_single_example(y_t, y_p, num_classes=self.num_classes, overlap=0.25))
        #         self.k_50.append(f1_at_k_single_example(y_t, y_p, num_classes=self.num_classes, overlap=0.5))

        # else:
        #     self.k_10.append(f1_at_k_single_example(y_true, y_pred, num_classes=self.num_classes, overlap=0.1))
        #     self.k_25.append(f1_at_k_single_example(y_true, y_pred, num_classes=self.num_classes, overlap=0.25))
        #     self.k_50.append(f1_at_k_single_example(y_true, y_pred, num_classes=self.num_classes, overlap=0.5))

    def compute(self):
        return np.mean(self.k_10), np.mean(self.k_25), np.mean(self.k_50)


def majority_voting(arr, total_len, buffer_len, weighted):
    # TODO: add asserts
    # majority voting for overlapping windows for single sample
    # arr: [T,]

    # Example:
    # arr = np.array([[1, 2, 3], 
    #                    [2, 3, 4], 
    #                       [3, 4, 5], 
    #                          [4, 5, 6]])
    # [1...6] <= majority_voting(arr)


    result = np.zeros((total_len), dtype=np.int32)
    confidences = np.zeros((total_len))
    shift = buffer_len - 1

    if weighted != "none":
        # w1 and w2 are only for buffer_len=30
        if weighted == "w1":
            vote_weights = np.array([0]*1 + [1]*1 + [2]*1 + [5]*2 + [6]*3 + [7]*4 + [8]*6 +[7]*4 + [6]*3 + [5]*2 + [2]*1 + [1]*1 + [0]*1)
        elif weighted == "w2":
            vote_weights = np.array([0]*3 + [4]*2 + [5]*3 + [6]*3 + [7]*3 + [8]*2 + [7]*3 +[6]*3 + [5]*3 + [4]*2 + [0]*3)
        elif weighted == "auto":
            # 4 ... 8 ... 4
            half = np.linspace(4, 8, buffer_len//2).round().astype(int)
            if buffer_len % 2 == 0:
                vote_weights = np.concatenate([half, half[::-1]])
            else:
                vote_weights = np.concatenate([half, [8], half[::-1]]) 
        else:
            raise ValueError("weighted must be one of 'none', 'w1', 'w2', or 'auto'")

        # 7//2 -> 3 but -7//2 -> -4
        assert (vote_weights[:buffer_len//2] == vote_weights[-(buffer_len//2):][::-1]).all(), "vote_weights is supposed to be symmetric"
        # assert vote_weights.shape[0] == 30, "vote_weights must have length 30"
        assert vote_weights.shape[0] == buffer_len, "vote_weights must have same length as buffer_len"
    
    # First element:
    result[0], confidences[0] = arr[0], 1
    
    for i in range(1, total_len):
        if i < buffer_len - 1: # First buffer_len-1 frames
            j = i
            repeat = i + 1 
        elif buffer_len - 1 <= i < total_len - buffer_len + 1: # Middle frames
            j = (i - shift) * buffer_len + shift
            repeat = buffer_len
        elif total_len - buffer_len + 1 <= i: # Last buffer_len-1 frames
            j = (i - shift) * buffer_len + shift
            repeat = total_len - i
        
        sub_arr = arr[j : j + repeat * (buffer_len - 1) : buffer_len - 1]

        assert sub_arr.shape[0] == repeat, f"{sub_arr.shape[0]} != {repeat}"

        if weighted != "none":
            if sub_arr.shape[0] == vote_weights.shape[0]: 
                sub_arr = np.repeat(sub_arr, vote_weights)


        # Count votes
        values, counts = np.unique(sub_arr, return_counts=True)
        ind = np.argmax(counts)
        elected = values[ind]

        result[i], confidences[i] = elected, counts[ind] / sub_arr.shape[0]

    return result, confidences


def combine_predictions(all_arr, idx, sub_arr, mapping):
    # first half of sub_arr is for left hand, second half is for right hand
    
    sub_arr = sub_arr.repeat(2, axis=1) # upsample, bc we have a tubelet shape of 16x16x2 (h, w, t) 

    double_bs = sub_arr.shape[0]
    assert 2*len(idx) == double_bs, f"2*idx and sub_arr must have same shape: {2*len(idx)} != {double_bs}"

    left_arr , right_arr = sub_arr[:double_bs//2, :], sub_arr[double_bs//2:, :]

    for half_arr, hand in zip([left_arr, right_arr], ["L", "R"]):
        for j in range(len(idx)):
            p = half_arr[j]
            i = idx[j]

            subject, task, take, start_frame, act = mapping[i]

            # end frame exclusive i.e. [start_frame, end_frame)
            start_frame = start_frame // STRIDE
            temp_len = 32 # FIXME: hardcoded

            all_arr[f"{subject}{task}{take}{hand}"][temp_len * start_frame : temp_len * (1+start_frame)] = p
    
    return all_arr

def calc_metrics_at_once(all_arr, mapping, num_classes, temporal_length, weighted_mv, save_dir=None):
    # Applies majority voting, calculates metrics, visualizes and saves the all predictions
    
    # In mapping there is no mirrored data, since it is redundant

    f1_at_k = F1_at_k(num_classes=num_classes)
    f1_classic = MulticlassF1Score(num_classes=num_classes, average='macro')
    conf_mat = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true")

        
    # if save_dir is not None:
    #     if os.path.exists(save_dir):
    #         shutil.rmtree(save_dir)
    #     os.makedirs(save_dir)

    for video_id, y_true_LR in mapping.items():

        y_true_L, y_true_R = y_true_LR[:, 0], y_true_LR[:, 1]  
        
        y_pred_L_all = all_arr[f"{video_id}L"]
        y_pred_R_all = all_arr[f"{video_id}R"]


        for y_pred_ in [y_pred_L_all, y_pred_R_all]:
            assert -1 not in y_pred_, "Not filled appropriately. There should not be -1 in y_pred"

        y_pred__L, conf_L = majority_voting(y_pred_L_all, total_len=y_true_L[::STRIDE].shape[0], buffer_len=temporal_length, weighted=weighted_mv)
        y_pred__R, conf_R = majority_voting(y_pred_R_all, total_len=y_true_R[::STRIDE].shape[0], buffer_len=temporal_length, weighted=weighted_mv)
        

        y_pred__L = y_pred__L.repeat(2, axis=0) # upsample
        y_pred__R = y_pred__R.repeat(2, axis=0) # upsample

        if len(y_true_L) - len(y_pred__L) == 0:
            pass
        elif len(y_true_L) - len(y_pred__L) == 1:
            y_true_L = y_true_L[:-1]
            y_true_R = y_true_R[:-1]
        elif len(y_pred__L) - len(y_true_L) == 1:
            y_pred__L = y_pred__L[:-1]
            y_pred__R = y_pred__R[:-1]
        else:
            print("#########")
            print("######### Lengths must be equal or one less", len(y_true_L), len(y_pred__L), video_id)
            continue
        
        # Notice that the convention is (y_true, y_pred)
        f1_at_k(y_true_L, y_pred__L)
        f1_at_k(y_true_R, y_pred__R)
                
        
        # Notice that the convention is (y_pred, y_true)
        f1_classic(torch.tensor(y_pred__L), torch.tensor(y_true_L))
        f1_classic(torch.tensor(y_pred__R), torch.tensor(y_true_R))

        # Conf mat
        # To ignore warnings about nan values in confusion matrix and functorch's performance drop
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*NaN values found in confusion matrix.*', UserWarning)

            conf_mat(torch.tensor(y_pred__L), torch.tensor(y_true_L))
            conf_mat(torch.tensor(y_pred__R), torch.tensor(y_true_R))        
        
        # Visualization
        # FIXME
        # if save_dir is not None:
        #     plot_segmentation_v2(y_true_L, y_pred__L, conf_L, y_pred_MR, conf_MR, save_file=opj(save_dir, video_id+"_L" + ".png"), title=video_id+"_L")
        #     plot_segmentation_v2(y_true_R, y_pred__R, conf_R, y_pred_ML, conf_ML, save_file=opj(save_dir, video_id+"_R" + ".png"), title=video_id+"_R")

    return *f1_at_k.compute(), f1_classic.compute().item(), conf_mat.compute().cpu().numpy()



def metrics_to_str(metrics):
    selected=["f1_top1", "f1_top3", "f1_top1_once", "f1_top3_once", "f1@10", "f1@25", "f1@50"]
    return " ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k in selected])
    # return " ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if "cm" not in k])

    

class VisIndividualPreds:
    def __init__(self, num_of_samples, training_len, val_len, save_dir) -> None:
        """To visualize individual predictions"""

        # [mode][idx][epoch] -> [y_pred]
        self.segments_pred = {"train": {}, "val": {}}

        # [mode][idx] -> [y_true]
        self.segments_gt = {"train": {}, "val": {}}

        self.num_of_samples = num_of_samples
        self.save_dir = save_dir
        
        self.selected_training_idx=np.random.choice(training_len, num_of_samples, replace=False)
        for s_idx in self.selected_training_idx:
            self.segments_pred["train"][s_idx] = {}


        self.selected_val_idx=np.random.choice(val_len, num_of_samples, replace=False)
        for s_idx in self.selected_val_idx:
            self.segments_pred["val"][s_idx] = {}

    def pick_samples(self, *, y_true, y_pred, idx, epoch, mode):
        # called in each epoch
        # pick samples for visualization
        # save them in self.segments_pred
        # expect y_true and y_pred to be of shape [temp len, batch size]

        assert y_true.shape == y_pred.shape, f"y_true and y_pred must have same shape: {y_true.shape} != {y_pred.shape}"
        assert len(y_true.shape) == 2, f"{y_true.shape} must be 2D"

        selected_idx=self.selected_training_idx if mode=="train" else self.selected_val_idx

        for s_idx in selected_idx:
            # check which selected_idx is in the idx, find mapping and slice it from y_true and y_pred
            if s_idx not in idx:
                continue
            else:
                idx_correspondence = idx.tolist().index(s_idx)

                # For debug purposes, check if there is one and only one assignment.
                if epoch not in self.segments_pred[mode][s_idx]:
                    self.segments_pred[mode][s_idx][epoch] = y_pred[:, idx_correspondence]
                else:
                    raise ValueError("There is more than one assignment")

                if s_idx not in self.segments_gt[mode]:
                    self.segments_gt[mode][s_idx] = y_true[:, idx_correspondence]
                else:
                    assert (self.segments_gt[mode][s_idx] == y_true[:, idx_correspondence]).all(), "Conflicting assignments for ground truth"
                    

    def save(self):
        # call this after loop is finished i.e. epoch == args.max_epochs
        # FIXME
        pass
        # for mode in ["train", "val"]:
        #     selected_idx=self.selected_training_idx if mode=="train" else self.selected_val_idx

        #     for s_idx in selected_idx:

        #         # check if there is an assignment
        #         plot_segmentation_partial(self.segments_pred[mode][s_idx], self.segments_gt[mode][s_idx], self.save_dir, f"{mode}_{s_idx}")



def print_warning(text, color=(230, 50, 10)):
    print(f"\033[38;2;{color[0]};{color[1]};{color[2]}m{text} \033[38;2;255;255;255m")
 


def get_loss_weights(weighted_loss_type, distribution):

    if weighted_loss_type == "none":
        loss_weights = None
    elif weighted_loss_type == "inv":
        loss_weights = 1.0 / torch.tensor(distribution)
        loss_weights = loss_weights / loss_weights.sum() * len(ACTIONS)
    elif weighted_loss_type == "sqrt_inv":
        loss_weights = 1.0 / torch.sqrt(torch.tensor(distribution))
        loss_weights = loss_weights / loss_weights.sum() * len(ACTIONS)
    elif weighted_loss_type == "effective":
        beta = 0.9999
        effective_num = 1.0 - torch.power(beta, distribution)
        loss_weights = (1.0 - beta) / effective_num
        loss_weights = loss_weights / loss_weights.sum() * len(ACTIONS)
    else:
        raise NotImplementedError(f"Weighted loss type {weighted_loss_type} is not implemented")

    return loss_weights


class Metrics:
    def __init__(self, args, num_classes, temp_len, device, gt_dict, sample_dict, mode) -> None:
        
        self.gt_dict = gt_dict
        self.sample_dict = sample_dict
        self.mode = mode
        self.temp_len = temp_len
        self.weighted_mv = args.weighted_mv 
        
        assert temp_len == 32, f"temp_len must be 32, but it is {temp_len}"


        # Window-based
        self.f1_top1 = MulticlassF1Score(num_classes=num_classes, top_k=1, average="macro").to(device)
        self.f1_top3 = MulticlassF1Score(num_classes=num_classes, top_k=1, average="macro").to(device) # top_k is intentionally 1

        
        # Window-location-sensitive
        self.win_f1_calc_top1 = [MulticlassF1Score(num_classes=num_classes, top_k=1, average="macro").to(device) for _ in range(self.temp_len//2)]
        self.win_f1_calc_top3 = [MulticlassF1Score(num_classes=num_classes, top_k=1, average="macro").to(device) for _ in range(self.temp_len//2)]


        # filter gt_dict

        # After majority voting
        # Dictionaries to store predictions for each sample
        # We will also calculate F1 and CM after MV
        self.all_predictions_combined_top1 = {vid + hand: - np.ones((act[::STRIDE].shape[0] - self.temp_len + 1) * self.temp_len) for vid, act in gt_dict[mode].items() for hand in ["L", "R"]}
        self.all_predictions_combined_top3 = {vid + hand: - np.ones((act[::STRIDE].shape[0] - self.temp_len + 1) * self.temp_len) for vid, act in gt_dict[mode].items() for hand in ["L", "R"]}
        

    def update(self, pred, gt, idxes):
        # gt: [B*2, T/2]
        # pred: [B*2, T/2, C]

        gt_flattened = gt.flatten(start_dim=0, end_dim=1)
        pred_flattened = pred.flatten(start_dim=0, end_dim=1)

        pred_top_3 = torch.argsort(pred_flattened, dim=-1, descending=True)[:, :3]
        pred_top_1 = pred_top_3[:, 0]

        # This magical function ...
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*There is a performance drop.*', UserWarning)
            # pred_top_3 = torch.where(torch_vmap(lambda t1, t2: torch.isin(t1, t2))(gt_flattened, pred_top_3), gt_flattened, pred_top_1)
            pred_top_3 = torch.where(torch.vmap(lambda t1, t2: torch.isin(t1, t2))(gt_flattened, pred_top_3), gt_flattened, pred_top_1)
        
        
        # torchmetric
        step_f1_top1 = self.f1_top1(pred_top_1, gt_flattened)
        step_f1_top3 = self.f1_top3(pred_top_3, gt_flattened)


        pred_top_1_unf = pred_top_1.unflatten(0, (-1, self.temp_len//2)).cpu().numpy()
        pred_top_3_unf = pred_top_3.unflatten(0, (-1, self.temp_len//2)).cpu().numpy()

        # Window-location-sensitive metrics
        for i in range(self.temp_len//2):
            self.win_f1_calc_top1[i](pred[:, i], gt[:, i])
            self.win_f1_calc_top3[i](pred[:, i], gt[:, i])

        # These are expects non-flattened tensors
        combine_predictions(self.all_predictions_combined_top1, idxes.numpy(), pred_top_1_unf, self.sample_dict[self.mode])
        combine_predictions(self.all_predictions_combined_top3, idxes.numpy(), pred_top_3_unf, self.sample_dict[self.mode])

        # Only these are matters (actually they don't but whatever)
        return step_f1_top1.cpu().item(), step_f1_top3.cpu().item()


    def compute(self):

        f1_at_10, f1_at_25, f1_at_50, f1_top1_mv, cm_top1_mv = calc_metrics_at_once(self.all_predictions_combined_top1, 
                                                                                    self.gt_dict[self.mode], 
                                                                                    num_classes=len(ACTIONS), 
                                                                                    temporal_length=self.temp_len,
                                                                                    weighted_mv=self.weighted_mv,
                                                                                    save_dir= None) # self.seg_save_dir

        _, _, _, f1_top3_mv, cm_top3_mv = calc_metrics_at_once(self.all_predictions_combined_top3, 
                                                               self.gt_dict[self.mode], 
                                                               num_classes=len(ACTIONS), 
                                                               temporal_length=self.temp_len,
                                                               weighted_mv=self.weighted_mv,
                                                               save_dir=None)

        self.win_f1_calc_top1_results = [f1_calc.compute().item() for f1_calc in self.win_f1_calc_top1]
        self.win_f1_calc_top3_results = [f1_calc.compute().item() for f1_calc in self.win_f1_calc_top3]

        ret = {
            "f1_top1": self.f1_top1.compute().item(),
            "f1_top3": self.f1_top3.compute().item(),
            "f1_top1_once": f1_top1_mv,
            "f1_top3_once": f1_top3_mv,
            "cm_top1_once": cm_top1_mv,
            "cm_top3_once": cm_top3_mv,
            "f1@10": f1_at_10,
            "f1@25": f1_at_25,
            "f1@50": f1_at_50,
            "win_f1_top1": self.win_f1_calc_top1_results,
            "win_f1_top3": self.win_f1_calc_top3_results,
        }
        
        return ret 


def almost_equal(a, b, eps=1e-6):
    return np.abs(a-b) < eps


def wandb_logging(wandb, metric_dict, epoch, mode):
    # Gathered all wandb logs here (except batch logs)
    # I think wabdb kind a have state, so I take it as an argument. But not tested otherwise.


    # Window-location-sensitive metrics: line plots
    lplt1=plot_line(metric_dict["win_f1_top1"], f"{mode}_f1_top1_by_t", do_return_plot=True)
    wandb.log({f"{mode}/f1_by_t": {'top1': wandb.Image(lplt1)}})
    lplt1.close()

    lplt3=plot_line(metric_dict["win_f1_top3"], f"{mode}_f1_top3_by_t", do_return_plot=True)
    wandb.log({f"{mode}/f1_by_t": {'top3': wandb.Image(lplt3)}})
    lplt3.close()

    wandb.log({f"{mode}/epoch": {"f1@10": metric_dict["f1@10"], 
                            "f1@25": metric_dict["f1@25"], 
                            "f1@50": metric_dict["f1@50"], 
                            "f1_top1_once": metric_dict["f1_top1_once"], 
                            "f1_top3_once": metric_dict["f1_top3_once"],
                            "f1_top1": metric_dict["f1_top1"],
                            "f1_top3": metric_dict["f1_top3"]},
                "epoch": epoch})



    # Confusion matrices (only for test and val)
    if mode in ["val", "test"]:
        plt = plot_conf_mat(metric_dict["cm_top1_once"], f"{mode}_CM_top1_e{epoch:0>4}", do_return_plot=True)
        wandb.log({f"{mode}/confusion_matrix": {'top1': wandb.Image(plt)}})
        plt.close()

        plt3 = plot_conf_mat(metric_dict["cm_top3_once"], f"{mode}_CM_top3_e{epoch:0>4}", do_return_plot=True)
        wandb.log({f"{mode}/confusion_matrix": {'top3': wandb.Image(plt3)}})
        plt3.close()
