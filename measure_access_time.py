import time
import numpy as np
from PIL import Image
from itertools import product

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

# access_dir1 = "/home/stannis/enes/bimanual/ds2_bimacs_rgbd_data"
access_dir1 = "/YEREL/ds2_bimacs_rgbd_data_debug"
access_dir2 = "/ari/users/eerdogan03/ds2_bimacs_rgbd_data_debug"

n_frames_per_vid = 150
n_frames = n_frames_per_vid * 2 * 2 * 2
read_idx=0

start_time = time.perf_counter()
for subject, task, take, f in product([1, 2], [1, 2], [1, 7], range(0, n_frames_per_vid, 2)):

    path = f"{access_dir1}/subject_{subject}/{TASKS[task-1]}/take_{take}/rgb/chunk_{f//100}/frame_{f}.png"
    img = np.array(Image.open(path))
    read_idx += 1

elapsed_time = time.perf_counter() - start_time
print(f"Elapsed time: {elapsed_time/n_frames}s")


start_time = time.perf_counter()
for subject, task, take, f in product([1, 2], [1, 2], [1, 7], range(0, n_frames_per_vid, 2)):

    path = f"{access_dir1}/subject_{subject}/{TASKS[task-1]}/take_{take}/rgb/chunk_{f//100}/frame_{f}.png"
    img = np.array(Image.open(path))
    read_idx += 1

elapsed_time = time.perf_counter() - start_time
print(f"Elapsed time: {elapsed_time/(n_frames):.2f}s")

print(f"read_idx: {read_idx}, n_frames: {n_frames}")