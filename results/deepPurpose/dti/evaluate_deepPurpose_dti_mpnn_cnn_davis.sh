#!/bin/bash
#SBATCH --job-name=evaluate_deepPurpose
#SBATCH --output evaluate_deepPurpose_dti_mpnn_cnn_davis_50epoch.log
#SBATCH --error error_evaluate_deepPurpose_dti_mpnn_cnn_davis.log
#SBATCH --partition gpu
#SBATCH --mem=64G      
#SBATCH --gres=gpu:1

echo "Date"
date

start_time=$(date +%s)

# Load the necessary modules
# module purge
module load cuda/12.3   # Change this to the appropriate CUDA version
# module load cudnn/8.0.4   # Change this to the appropriate cuDNN version
# module load anaconda/2020.11  # Change this to the appropriate Anaconda version

# pip install --user DeepPurpose --quiet
# pip install --user git+https://github.com/bp-kelley/descriptastorus --quiet

# Activate your Python environment
source activate /home/debnathk/articles/.venv/bin/python

# Run your Python script
python evaluate_deepPurpose_dti_mpnn_cnn_davis.py

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"