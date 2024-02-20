#!/bin/bash
#SBATCH --job-name=deepDrug
#SBATCH --output output_deepDrug_dti_kiba.log
#SBATCH --error error_deepDrug_dti_kiba.log
#SBATCH --partition gpu
#SBATCH --mem=128G           # Set the memory required per node
#SBATCH --gres=gpu:2    # Request 2 GPU

echo "Date"
date

start_time=$(date +%s)

# Load the necessary modules
# module purge
module load cuda/12.3   # Change this to the appropriate CUDA version

# Activate your Python environment
# source activate your_conda_environment  # Change this to the name of your Conda environment

# Navigate to the directory containing your Python script
# cd /path/to/your/python/script

# Run your Python script
python deepdrug.py --configfile ./config/KIBA.regression.yml

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"