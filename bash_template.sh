#!/bin/bash
#SBATCH --job-name=training_test
#SBATCH --time=12:00:00
#SBATCH --partition=shared-gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --mem=48GB

### Remove limit of files for training
ulimit -n 65535

# Try without
# module load CUDA/11.8.0
# module load cuDNN/8.6.0.163-CUDA-11.8.0

apptainer exec --nv \
  --bind /usr/local/cuda/targets/x86_64-linux:/usr/local/cuda/targets/x86_64-linux \
  --bind /usr/local/cuda/lib64:/usr/local/cuda/lib64 \
  --bind /srv/beegfs/scratch/shares/upeguipa/SST1M \
  ctlearnenv.sif bash -c '
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
python3 scripts/train_model.py --yaml_file /home/users/v/varenneh/configs/type_config.yaml
'

