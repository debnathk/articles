#!/bin/bash
#SBATCH --job-name=graphDTA
#SBATCH --output output_deepPurpose_dti_kiba.log
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem=128G           # Set the memory required per node
#SBATCH --gres=gpu:1    # Request 1 GPU
#SBATCH --time=72:00:00    # Set the maximum time your job will run (hh:mm:ss)

echo "Date"
date

start_time=$(date +%s)

# Load the necessary modules
# module purge
# module load cuda/12.3   # Change this to the appropriate CUDA version
# module load cudnn/8.0.4   # Change this to the appropriate cuDNN version
# module load anaconda/2020.11  # Change this to the appropriate Anaconda version

# Creating directories
mkdir -p ./test_models
echo 'Folder created: test_models'

mkdir -p ./test_results
echo 'Folder created: test_results'

# Activate your Python environment
# source activate your_conda_environment  # Change this to the name of your Conda environment

# Navigate to the directory containing your Python script
# cd /path/to/your/python/script

# Run your Python script
# python getFeatures.py example_params.txt
# python cvae.py --dir dataset -a 6 -b 0.1 -m 300 --save --layer 1000 100
# python cvae.py --dir dataset --rating -a 15 -b 3 -m 500 --load 1 --layer 1000 100

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"