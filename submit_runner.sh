#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J jpeg_comp_99_fix
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
## -- I want a gpu... (nvm, we don't have any) --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need 2GB of memory per core/slot (might still be a bit overkill-- 
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 3GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 6:00
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s204084@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o /dtu/blackhole/1d/155421/sidbench/bsub/Output_%J.out 
#BSUB -eo /dtu/blackhole/1d/155421/sidbench/bsub/Output_%J.err

cd /dtu/blackhole/1d/155421/
source .venv_dl16_mpi/bin/activate
python3 --version

cd sidbench

python3 bsub_runner.py --masking --batchSize=128 --jpegcompress_quality=100