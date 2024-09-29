#!/bin/bash

# Default values
gen_sample_size=""
output_dir=""

# Parse input parameters passed to the script
for ARGUMENT in "$@"; do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
        --num_gen_samples) gen_sample_size=${VALUE} ;;
        --ckpt_dir) output_dir=${VALUE} ;;
        *) echo "Unknown argument: $KEY" ;;
    esac
done

# Check that required parameters are set
if [ -z "$gen_sample_size" ] || [ -z "$output_dir" ]; then
    echo "Error: Required parameters --gen_sample_size or --output_dir are missing"
    exit 1
fi

# Check if OpenJDK is available in the environment
if command -v java &> /dev/null && java -version 2>&1 | grep -q "openjdk"; then
    echo "OpenJDK is already available in the environment."
else
    echo "OpenJDK is not available in the environment."
    module load openjdk/18.0.1.1
    echo "OpenJDK 18.0.1.1 has been loaded."
fi

# Construct the Java command dynamically
COMMAND="java -jar synthea-with-dependencies.jar"
COMMAND+=" -p $gen_sample_size"
COMMAND+=" --exporter.csv.export=true"
COMMAND+=" --exporter.baseDirectory=$output_dir"

# Output the command (for debugging)
echo "Executing command: $COMMAND"

# Execute the Java command
$COMMAND

# java -jar synthea-with-dependencies.jar -p 1000 --exporter.csv.export=true --exporter.baseDirectory="/scratch/zhenkewu_root/zhenkewu1/chenxran/syntheticEHR/results/synthea"