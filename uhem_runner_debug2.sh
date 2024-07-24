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

eval "$(/ari/users/eerdogan03/miniconda3/bin/conda shell.bash hook)"
conda activate pyg_env

python -c "import torch;print('is_available:', torch.cuda.is_available());import os;print('ls:', os.listdir());devices = [d for d in range(torch.cuda.device_count())];device_names  = [torch.cuda.get_device_name(d) for d in devices];print('device_names:', device_names);"

nvidia-smi