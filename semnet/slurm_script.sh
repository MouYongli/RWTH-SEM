#!/usr/local_rwth/bin/zsh

### ask for 10 GB memory
#SBATCH --mem-per-cpu=12040M   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
### name the job
#SBATCH --job-name=SEM
### job run time
#SBATCH --time=25:00:00
### declare the merged STDOUT/STDERR file
#SBATCH --output=output.%J.txt
###
#SBATCH --mail-type=ALL
###
#SBATCH --mail-user=yongli.mou@rwth-aachen.de
### request a GPU
#SBATCH --gres=gpu:pascal:1


### begin of executable commands
cd $HOME/SEMNET/semnet
### load modules
module switch intel gcc
module load python/3.6.8
module load cuda/92
module load cudnn/7.4
python3 main.py