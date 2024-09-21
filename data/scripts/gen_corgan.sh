#!/bin/bash

# Default values
gen_sample_size=""
DATASETPATH=""
expPATH=""
training="True"
generate="True"

# Parse input parameters passed to the script
for ARGUMENT in "$@"; do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
        --num_gen_samples) gen_sample_size=${VALUE} ;;
        --real_data_path) DATASETPATH=${VALUE} ;;
        --ckpt_dir) expPATH=${VALUE} ;;
        *) echo "Unknown argument: $KEY" ;;
    esac
done

# Check that required parameters are set
if [ -z "$DATASETPATH" ] || [ -z "$expPATH" ] || [ -z "$gen_sample_size" ]; then
    echo "Error: Required parameters --DATASETPATH or --expPATH or --gen_sample_size are missing."
    exit 1
fi

# Construct the Python command dynamically
COMMAND="python /home/chenxran/projects/synthEHRella/synthEHRella/data/methods/cor-gan/Generative/corGAN/pytorch/CNN/MIMIC/wgancnnmimic.py"
COMMAND+=" --DATASETPATH $DATASETPATH"
COMMAND+=" --expPATH $expPATH"
COMMAND+=" --gen_sample_size $gen_sample_size"
COMMAND+=" --training $training"
COMMAND+=" --generate $generate"

# Output the command (for debugging)
echo "Executing command: $COMMAND"

# Execute the Python command
$COMMAND
