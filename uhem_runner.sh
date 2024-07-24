#!/bin/bash
#SBATCH -A gtat01                # account / proje adi
#SBATCH -p a100q
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --mail-type=END,FAIL                     # is bitince yada sorun cikarsa mail at
#SBATCH --mail-user=erdogane16@itu.edu.tr        # mail atilacak adres
#SBATCH -o deneme.out      
#SBATCH -e deneme.err
#SBATCH --gres=gpu:1
#SBATCH -J vivit_deneme

# Data ve kodu kopyala
cp -r /ari/users/eerdogan03/ds2_bimacs_rgbd_data /YEREL
cp -r /ari/users/eerdogan03/bimacs_rgbd_data_ground_truth /YEREL
cp -r /ari/users/eerdogan03/ViViT-Bimanual /YEREL

eval "$(/ari/users/eerdogan03/miniconda3/bin/conda shell.bash hook)"
conda activate pyg_env


# chmod +x /YEREL/ViViT-Bimanual/train0.sh
# ./ViViT-Bimanual/train0.sh
python /YEREL/ViViT-Bimanual/main.py -n sgd_lr4 -o SGD -lr 0.0001 -mn 0.9 -wd 0 --num_epochs 4 --num_acc 32 -bs 4

#Program bitince olusan ve saklamak istediklerimizi geri ev dizinine kopyalamali
cp -r /YEREL/ViViT-Bimanual/saved_models /ari/users/eerdogan03/