#!/bin/bash
#SBATCH --job-name=run_graphRepur
#SBATCH --output output_graphRepur.log
#SBATCH --error error_graphRepur.log
#SBATCH --partition gpu
#SBATCH --mem=64G           # Set the memory required per node
#SBATCH --gres=gpu:1    # Request 1 GPU

echo "Date"
date

start_time=$(date +%s)

# Load the necessary modules
# module purge
module load cuda/12.3   # Change this to the appropriate CUDA version
# module load cudnn/8.0.4   # Change this to the appropriate cuDNN version
# module load anaconda/2020.11  # Change this to the appropriate Anaconda version

# Activate your Python environment
source activate /home/debnathk/articles/.venv/bin/python

# Run your Python script
python pre_trained_GraphRepur.py

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"