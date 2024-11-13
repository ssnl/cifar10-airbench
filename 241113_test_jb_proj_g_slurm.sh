#!/bin/bash

set +e

START_TIME=$(date +"%m/%d/%y-%H:%M")
UUID6=$(echo -n $(date +%s%N) | md5sum | fold -w6 | shuf | head -n1)

# Define commands as a multi-line string
commands_str=$(cat << 'EOF'

python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.001_final_scale0.001 -- python -u 241113_test_jb_proj_g.py 0.001 0.001 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.00231013_final_scale0.001 -- python -u 241113_test_jb_proj_g.py 0.001 0.00231013 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.0053367_final_scale0.001 -- python -u 241113_test_jb_proj_g.py 0.001 0.0053367 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.0123285_final_scale0.001 -- python -u 241113_test_jb_proj_g.py 0.001 0.0123285 0
EOF
)

# Convert the multi-line string into an array, splitting by newline
IFS=$'\n' read -rd '' -a commands <<< "$commands_str"

# Set up progress bar variables
total_commands=${#commands[@]}
echo "total_commands=$total_commands"
bar_width=50  # Width of the progress bar
count=0

# Loop over each command in the array
for cmd in "${commands[@]}"; do
    eval "$cmd"  # Execute the command
    ((count++))

    # Calculate progress percentage and the number of hashes for the bar
    progress=$((count * 100 / total_commands))
    num_hashes=$((count * bar_width / total_commands))
    bar=$(printf "%-${bar_width}s" "#" | cut -c 1-$num_hashes)

    # Display the progress bar with the percentage
    echo -ne "Progress: [$bar] $progress% \r"
done

# Move to a new line after the progress bar completes
echo -e "\nAll commands completed."

echo "START_TIME=$START_TIME"

