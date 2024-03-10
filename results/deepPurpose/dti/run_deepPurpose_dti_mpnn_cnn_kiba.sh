#!/bin/bash
#SBATCH --job-name=deepPurpose
#SBATCH --output output_deepPurpose_dti_kiba.log
#SBATCH --partition gpu
#SBATCH --mem=128G           # Set the memory required per node
#SBATCH --gres=gpu:1         # Request 1 GPU

echo "Date"
date

start_time=$(date +%s)

# Load the necessary modules
module load cuda/12.3        # Change this to the appropriate CUDA version

# pip install --user DeepPurpose
# pip install --user git+https://github.com/bp-kelley/descriptastorus

# Run your Python script
python run_deepPurpose_dti_mpnn_cnn_kiba.py

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"