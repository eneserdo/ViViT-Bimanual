#!/bin/bash
python main.py -n rmsprop_lr1_wd4 -o RMSprop -lr 0.1 -wd 0.0001 --num_epochs 10 --num_acc 32 --gpu_id 1
sleep 5

python main.py -n rmsprop_lr2 -o RMSprop -lr 0.01 -wd 0 --num_epochs 10 --num_acc 32 --gpu_id 1
sleep 5