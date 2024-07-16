#!\bin\bash
python main.py -n sgd_lr1_wd4 -o SGD -lr 0.1 -mn 0.9 -wd 0.0001 --num_epochs 20 --num_acc 32 --gpu_id 0
sleep 5

python main.py -n sgd_lr2 -o SGD -lr 0.01 -mn 0.9 -wd 0 --num_epochs 10 --num_acc 32 --gpu_id 0
sleep 5

