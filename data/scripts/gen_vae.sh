#!/bin/bash

# Default values
DATASETPATH=""
MODELPATH=""
num_fake_samples=""
train="True"

# Parse input parameters passed to the script
for ARGUMENT in "$@"; do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
        --real_data_path) DATASETPATH=${VALUE} ;;
        --ckpt_dir) MODELPATH=${VALUE} ;;
        --num_gen_samples) num_fake_samples=${VALUE} ;;
        *) echo "Unknown argument: $KEY" ;;
    esac
done

# Check that required parameters are set
if [ -z "$DATASETPATH" ] || [ -z "$MODELPATH" ] || [ -z "$num_fake_samples" ]; then
    echo "Error: Required parameters --DATASETPATH or --PATH or --num_fake_samples are missing."
    exit 1
fi

# Construct the Python command dynamically
COMMAND="python /home/chenxran/projects/synthEHRella/synthEHRella/data/methods/cor-gan/Generative/VAE/MIMIC/vaeConvolutional.py"
COMMAND+=" --DATASETPATH $DATASETPATH"
COMMAND+=" --MODELPATH $MODELPATH"
COMMAND+=" --num_fake_samples $num_fake_samples"
COMMAND+=" --train $train"

# Output the command (for debugging)
echo "Executing command: $COMMAND"

# Execute the Python command
$COMMAND
