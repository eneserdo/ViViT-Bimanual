#!/bin/bash
#SBATCH -A gtat01                # account / proje adi
#SBATCH -p a100q
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --mail-type=END,FAIL                     # is bitince yada sorun cikarsa mail at
#SBATCH --mail-user=erdogane16@itu.edu.tr        # mail atilacak adres
#SBATCH -o m2_1_d.out      
#SBATCH -e m2_1_d.err
#SBATCH --gres=gpu:1
#SBATCH -J vivit_m2

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

# python main.py -n sgd_lr4 -o SGD -lr 0.0001 -mn 0.9 -wd 0 --num_epochs 20 --num_acc 32 -bs 2
# python main.py -n M2_sgd_l3_h8 --model tr --num_workers 2 -o SGD -lr 0.001 -bs 32 --num_acc 16 --nhead 8 --num_layers 3 --dim_feedforward 2048 --base_lr 0.0005 --num_epochs 5
# sleep 5

python main.py -n M2_sgd_l3_h8 --model tr --num_workers 4 -o SGD -lr 0.001 -bs 64 --num_acc 16 --nhead 8 --num_layers 3 --dim_feedforward 2048 --base_lr 0.0005 --num_epochs 1 --use_randaug
sleep 5

#Program bitince olusan ve saklamak istediklerimizi geri ev dizinine kopyalamali
cp -r saved_models /ari/users/eerdogan03/