#!/bin/bash

# Default values
num_fake_samples=""
DATASETPATH=""
PATH=""
training="True"

# Parse input parameters passed to the script
for ARGUMENT in "$@"; do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
        --num_gen_samples) num_fake_samples=${VALUE} ;;
        --real_data_path) DATASETPATH=${VALUE} ;;
        --ckpt_dir) PATH=${VALUE} ;;
        *) echo "Unknown argument: $KEY" ;;
    esac
done

# Check that required parameters are set
if [ -z "$DATASETPATH" ] || [ -z "$PATH" ] || [ -z "$num_fake_samples" ]; then
    echo "Error: Required parameters --DATASETPATH or --PATH or --num_fake_samples are missing"
    exit 1
fi

# Construct the Python command dynamically
COMMAND="python /home/chenxran/projects/synthEHRella/synthEHRella/data/methods/cor-gan/Generative/medGAN/MIMIC/pytorch/MLP/medGAN.py"
COMMAND+=" --DATASETPATH $DATASETPATH"
COMMAND+=" --PATH $PATH"
COMMAND+=" --num_fake_samples $num_fake_samples"
COMMAND+=" --training $training"

# Output the command (for debugging)
echo "Executing command: $COMMAND"

# Execute the Python command
$COMMAND
