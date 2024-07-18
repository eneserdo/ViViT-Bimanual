#!/bin/bash
python main.py -n sgd_lr4 -o SGD -lr 0.0001 -mn 0.9 -wd 0 --num_epochs 10 --num_acc 32 --gpu_id 0
sleep 5

python main.py -n sgd_lr4_c1 -o SGD -lr 0.0001 -mn 0.9 -wd 0 --num_epochs 10 --num_acc 32 --gpu_id 0 --clip 1
sleep 5

