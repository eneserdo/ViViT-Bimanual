#!/bin/bash
#SBATCH -A gtat01                # account / proje adi
#SBATCH -p a100q
#SBATCH -N 1
#SBATCH -n 64 ???
#SBATCH --mail-type=END,FAIL                     # is bitince yada sorun cikarsa mail at
#SBATCH --mail-user=erdogane16@itu.edu.tr        # mail atilacak adres

#SBATCH --gres=gpu:1 ???

#Girdi dosyalari kopyalaniyor
\cp -r /ari/users/eerdogan03/ds2_bimacs_rgbd_data /YEREL
\cp -r /ari/users/eerdogan03/bimacs_rgbd_data_ground_truth /YEREL
# \cp -r /ari/users/eerdogan03/ViViT-Bimanual /YEREL ???

# Create env?

chmod +x /YEREL/ViViT-Bimanual/train0.sh
./ViViT-Bimanual/train0.sh

# Calistirmak istedigimiz programa uygun parametrelerle /YEREL diskini kullanmasi soylenmeli
# Tabiki kullanacaginiz programin parametreleri degisik olacaktir


#Program bitince olusan ve saklamak istediklerimizi geri ev dizinine kopyalamali
\cp -r /YEREL/ViViT-Bimanual/saved_models /ari/users/eerdogan03/saved_models