#!/bin/bash

# Define the start time
start_time=$(date +%s)

# Run the Python script
python gen_promptEHR.py

# Define the end time
end_time=$(date +%s)

# Calculate the elapsed time
elapsed_time=$((end_time - start_time))

# Display the running time
echo "The script took $elapsed_time seconds to run."
