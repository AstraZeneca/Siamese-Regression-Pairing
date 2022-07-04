#!/bin/bash
#SBATCH --job-name=train                              # job name
#SBATCH --output=train.txt   # write STDOUT to this file
#SBATCH --time=10:00:00                                                 # wall time dd-hh:mm:ss (dd = days, hh = hours, mm = minutes, ss = seconds)
#SBATCH --partition=gpu                                               # -p, set the cluster partition to gpu
#SBATCH --nodes=1                                                       # -N, node count
#SBATCH --cpus-per-task=4                                           # -c, cpu core count
#SBATCH --gres=gpu:2                                                   # specify use of a generic resource (single gpu)
#SBATCH --mem=20g                                                     # RAM requirement
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=noname19980927@gmail.com
cd ..
python mlp.py -s freesolv_top100.yml -f freesolv -st 100