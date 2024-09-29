#!/bin/bash

# Default values
real_data_path=""
gen_sample_size=""
output_data_path=""

# Parse input parameters passed to the script
for ARGUMENT in "$@"; do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
        --real_data_path) real_data_path=${VALUE} ;;
        --num_gen_samples) gen_sample_size=${VALUE} ;;
        --ckpt_dir) output_dir=${VALUE} ;;
        *) echo "Unknown argument: $KEY" ;;
    esac
done

# Check that required parameters are set
if [ -z "$gen_sample_size" ] || [ -z "$output_dir" ] || [ -z "$real_data_path" ]; then
    echo "Error: Required parameters --real_data_path or --num_gen_samples or --output_dir are missing."
    exit 1
fi

# Construct the Java command dynamically
COMMAND="Rscript plasmode.R"
COMMAND+=" $real_data_path $gen_sample_size $output_dir"

# Output the command (for debugging)
echo "Executing command: $COMMAND"

# Execute the Java command
$COMMAND