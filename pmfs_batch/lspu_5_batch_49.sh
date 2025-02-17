#!/usr/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH -JEV-spatial-matching         # job name
#SBATCH -A gts-smaguluri3      # account to which job is charged, ex: GT-gburdell3
#SBATCH -qinferno                 # queue name
#SBATCH -N1 --ntasks-per-node=1 -c1
#SBATCH --mem-per-cpu=8G                # memory per core
#SBATCH -t72:00:00
#SBATCH -ooutput_files/report-%j.out                     # combine output and error messages into 1 file

cd $SLURM_SUBMIT_DIR

module load anaconda3/2022.05.0.1
module load gurobi
conda init --all
source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate spatial-match

srun python /storage/home/hcoda1/7/vsivaraman3/p-smaguluri3-0/spatial-matching-sim-vishal/pmf_batch/lspu_5_batch_49.py

source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda deactivate
EOT
