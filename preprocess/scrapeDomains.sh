#!/bin/bash
#SBATCH --partition=cellbio-dgx # partition
#SBATCH --account=soderlinglab
#SBATCH --job-name=scrapeDomains    # job -name , change from command line 
#SBATCH --mem=200G
#SBATCH --time=12-00:00:00 # you have asked for 12 days
#SBATCH --output=slurm_output/%x.%j.out # Standard output log, %x is the job name, %j is the job ID, y is custom time stamp
#SBATCH --error=slurm_output/%x.%j.err      # Standard error log, %x is the job name, %j is the job ID
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pooja.parameswaran@duke.edu

echo Job Name: scrapeDomains


export TORCH_HOME='/cwork/pkp14'
python protein_info.py "$@"
