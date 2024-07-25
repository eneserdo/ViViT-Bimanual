#!/bin/bash
#SBATCH -A gtat01                # account / proje adi
#SBATCH -p a100q
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --mail-type=END,FAIL                     # is bitince yada sorun cikarsa mail at
#SBATCH --mail-user=erdogane16@itu.edu.tr        # mail atilacak adres
#SBATCH -o measure.out      
#SBATCH -e measure.err
#SBATCH --gres=gpu:1
#SBATCH -J access_measure

# Data ve kodu kopyala
cp -r /ari/users/eerdogan03/ds2_bimacs_rgbd_data_debug /YEREL
cp -r /ari/users/eerdogan03/ViViT-Bimanual /YEREL

echo "### copy done"

eval "$(/ari/users/eerdogan03/miniconda3/bin/conda shell.bash hook)"
conda activate pyg_env

echo "### from YEREL"

python /YEREL/ViViT-Bimanual/measure_access_time.py
sleep 5

echo "### from ari/users"

python /ari/users/eerdogan03/ViViT-Bimanual/measure_access_time.py
sleep 5