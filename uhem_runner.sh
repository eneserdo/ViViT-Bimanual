#!/bin/bash
#SBATCH -A gtat01                # account / proje adi
#SBATCH -p gpu2dq
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --mail-type=END,FAIL                     # is bitince yada sorun cikarsa mail at
#SBATCH --mail-user=erdogane16@itu.edu.tr        # mail atilacak adres
#SBATCH -o F48_S1_SGDN_RAC_wd_dr_m99_wlsi.out      
#SBATCH -e F48_S1_SGDN_RAC_wd_dr_m99_wlsi.err
#SBATCH --gres=gpu:1
#SBATCH -J F48_S1_SGDN_RAC_wd_dr_m99_wlsi

# Data ve kodu kopyala
# cp -r /ari/users/eerdogan03/ds2_bimacs_rgbd_data /YEREL
# cp -r /ari/users/eerdogan03/bimacs_rgbd_data_ground_truth /YEREL
cp -r /ari/users/eerdogan03/ViViT-Bimanual /YEREL

# echo "Done copying"

eval "$(/ari/users/eerdogan03/miniconda3/bin/conda shell.bash hook)"
conda activate pyg_env

cd /YEREL/ViViT-Bimanual

# chmod +x /YEREL/ViViT-Bimanual/train0.sh
# ./ViViT-Bimanual/train0.sh
# python measure_access_time.py
# sleep 5

# python main.py -n F_28_S0_Adam --num_workers 4 -o Adam --spatial_lr 0 --temporal_lr 0.001 -bs 32 --num_acc 8 --nhead 8 --num_layers 2 --dim_feedforward 2048 --num_epochs 10
# sleep 5

# python main.py -n F_28_S0 --num_workers 4 -o SGD --spatial_lr 0 --temporal_lr 0.001 -bs 32 --num_acc 8 --nhead 8 --num_layers 2 --dim_feedforward 2048 --num_epochs 10
# sleep 5

# python main.py -n F14_S0_Adam --num_workers 4 -o Adam --spatial_lr 0 --temporal_lr 0.001 -bs 32 --num_acc 8 --nhead 4 --num_layers 1 --dim_feedforward 1024 --num_epochs 10
# sleep 5

python main.py -n F48_S1_SGDN_RAC_wd_dr_m99_wlsi --num_workers 3 -o SGD --spatial_lr 0.01 --temporal_lr 0.01 -bs 32 --num_acc 8 --nhead 8 --num_layers 4 --dim_feedforward 2048 --num_epochs 13 --nesterov --use_randaug --clip 1 --dropout 0.2 --weight_decay 0.0001 --scheduler_step_size 1 3 5 7 8 10 --momentum 0.99 --weighted_loss sqrt_inv

sleep 5

cp -r saved_models /ari/users/eerdogan03/
