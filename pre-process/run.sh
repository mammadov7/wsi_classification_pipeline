
#!/bin/sh

#SBATCH --output=patch.out
#SBATCH --error=patch.err
#SBATCH --nodes=1
#SBATCH --partition=V100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem 40GB
#SBATCH --time=24:00:00


conda activate torch
which python
cd /home/ids/ipp-6635/wsi_patcher

python create_patches.py --source /tsi/medical/Ali/competition/part8 --step_size 256 --patch_size 256 --patch --save_dir /tsi/medical/Ali/train/part8 --seg --patch_level 2

pwd


python create_patches.py --source /gpfswork/rech/rnz/uyc98hc/Camelyon/normal --step_size 256 --patch_size 256 --patch --save_dir /gpfswork/rech/rnz/uyc98hc/Camelyon/h5/camelyon256/normal --seg --patch_level 2
python create_patches.py --source /gpfswork/rech/rnz/uyc98hc/Camelyon/test --step_size 256 --patch_size 256 --patch --save_dir /gpfswork/rech/rnz/uyc98hc/Camelyon/h5/camelyon256/test --seg --patch_level 2
python create_patches.py --source /gpfswork/rech/rnz/uyc98hc/Camelyon/tumor --step_size 256 --patch_size 256 --patch --save_dir /gpfswork/rech/rnz/uyc98hc/Camelyon/h5/camelyon256/tumor --seg --patch_level 2

module load pytorch-gpu/py3/1.11.0
module load openslide/3.4.1
cd $WORK
cd wsi_patcher

ssh jean-zay


srun -A rnz@cpu --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --time=12:00:00 --hint=nomultithread bash


python h5_bag.py --src /gpfswork/rech/rnz/uyc98hc/Camelyon/h5/camelyon224/normal/patches --dest /gpfsscratch/rech/rnz/uyc98hc/camelyon224/normal
python h5_bag.py --src /gpfswork/rech/rnz/uyc98hc/Camelyon/h5/camelyon224/test/patches --dest /gpfsscratch/rech/rnz/uyc98hc/camelyon224/test
python h5_bag.py --src /gpfswork/rech/rnz/uyc98hc/Camelyon/h5/camelyon224/tumor/patches --dest /gpfsscratch/rech/rnz/uyc98hc/camelyon224/tumor
python h5_bag.py --src /gpfswork/rech/rnz/uyc98hc/Camelyon/h5/camelyon256/normal/patches --dest /gpfsscratch/rech/rnz/uyc98hc/camelyon256/normal
python h5_bag.py --src /gpfswork/rech/rnz/uyc98hc/Camelyon/h5/camelyon256/test/patches --dest /gpfsscratch/rech/rnz/uyc98hc/camelyon256/test
python h5_bag.py --src /gpfswork/rech/rnz/uyc98hc/Camelyon/h5/camelyon256/tumor/patches --dest /gpfsscratch/rech/rnz/uyc98hc/camelyon256/tumor

# python create_patches.py \
#     --source /path/to/source/directory/with/slides/ \
#     --step_size 256 \
#     --patch_size 256 \
#     --patch \
#     --save_dir /path/to/output/ \
#     --seg \
#     --patch_level 2


