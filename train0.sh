#!/bin/bash

python main.py -n sgd_a_28 --model tr --num_workers 4 -o SGD -lr 0.001 --base_lr 0.0005 -bs 4 --num_acc 128 --nhead 8 --num_layers 2 --dim_feedforward 2048 --num_epochs 10
sleep 5

python main.py -n sgd_b_38 --model tr --num_workers 4 -o SGD -lr 0.001 -bs 4 --num_acc 128 --nhead 8 --num_layers 3 --dim_feedforward 2048 --num_epochs 10
sleep 5
