#!/bin/bash

set +e

START_TIME=$(date +"%m/%d/%y-%H:%M")
UUID6=$(echo -n $(date +%s%N) | md5sum | sed 's/.\{5\}$//' | fold -w6 | shuf | head -n1)

# Define commands as a multi-line string
commands_str=$(cat << 'EOF'
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr2.15443e-07_final_scale1e-06 -- python -u 241113_test_jb_proj_g.py 2.15443e-07 1e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr4.64159e-07_final_scale1e-06 -- python -u 241113_test_jb_proj_g.py 4.64159e-07 1e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr1e-06_final_scale1e-06 -- python -u 241113_test_jb_proj_g.py 1e-06 1e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr2.15443e-06_final_scale1e-06 -- python -u 241113_test_jb_proj_g.py 2.15443e-06 1e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr2.15443e-05_final_scale1e-06 -- python -u 241113_test_jb_proj_g.py 2.15443e-05 1e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr4.64159e-05_final_scale1e-06 -- python -u 241113_test_jb_proj_g.py 4.64159e-05 1e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.0464159_final_scale1e-06 -- python -u 241113_test_jb_proj_g.py 0.0464159 1e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr1e-07_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 1e-07 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr2.15443e-07_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 2.15443e-07 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr4.64159e-07_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 4.64159e-07 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr4.64159e-06_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 4.64159e-06 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr1e-05_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 1e-05 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr2.15443e-05_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 2.15443e-05 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr4.64159e-05_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 4.64159e-05 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.0001_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 0.0001 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.000215443_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 0.000215443 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.000464159_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 0.000464159 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.00464159_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 0.00464159 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.01_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 0.01 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.0215443_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 0.0215443 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.0464159_final_scale2.15443e-06 -- python -u 241113_test_jb_proj_g.py 0.0464159 2.15443e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr1e-07_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 1e-07 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr2.15443e-07_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 2.15443e-07 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr4.64159e-07_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 4.64159e-07 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr1e-06_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 1e-06 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr2.15443e-06_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 2.15443e-06 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr4.64159e-06_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 4.64159e-06 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr1e-05_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 1e-05 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr2.15443e-05_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 2.15443e-05 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr4.64159e-05_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 4.64159e-05 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.0001_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 0.0001 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.000215443_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 0.000215443 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.000464159_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 0.000464159 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.00464159_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 0.00464159 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.01_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 0.01 4.64159e-06 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation nvidia_geforce_rtx_2080_ti -uuid $UUID6 -q -e py312 proj_g_lr0.0464159_final_scale4.64159e-06 -- python -u 241113_test_jb_proj_g.py 0.0464159 4.64159e-06 0
EOF
)

# Convert the multi-line string into an array, splitting by newline
IFS=$'\n' read -rd '' -a commands <<< "$commands_str"

# Set up progress bar variables
total_commands=${#commands[@]}
bar_width=120  # Width of the progress bar
count=0
failed_commands=()  # Array to track failed commands
fail_count=0  # Counter to track number of failed commands

# Loop over each command in the array
for cmd in "${commands[@]}"; do
    eval "$cmd"  # Execute the command
    exit_code=$?  # Capture the exit code

    if [ $exit_code -ne 0 ]; then
        failed_commands+=("$cmd")  # Add to failed commands if it failed
        ((fail_count++))           # Increment the fail count
    fi

    ((count++))

    # Calculate progress percentage and the number of hashes for the bar
    progress=$((count * 100 / total_commands))
    num_hashes=$((count * bar_width / total_commands))

    # Create the progress bar with the correct number of # and spaces
    bar=$(printf "%-${bar_width}s" "$(printf '#%.0s' $(seq 1 $num_hashes))")

    # Display the progress bar with percentage, count, and failure count
    echo -ne "Progress: [${bar}] $progress% ($count/$total_commands, Failures: $fail_count) \r"
done
echo ""


# Print failed commands if there are any
if [ ${#failed_commands[@]} -ne 0 ]; then
    echo -e "\nThe following commands failed:"
    for failed_cmd in "${failed_commands[@]}"; do
        echo "  $failed_cmd"
    done
else
    echo "All commands executed successfully."
fi

echo "----------------------------------------"
echo "START_TIME=$START_TIME"
echo "UUID=$UUID6"
echo "----------------------------------------"


# Print Python snippet for status checking
cat << EOF

START_TIME = "$START_TIME"  # Set from bash variable
UUID = "$UUID6"  # Set from bash variable

import json, subprocess
import datetime, time
from IPython.display import display
from time import sleep


class printed_str(str):
    def __repr__(self):
       return self

dh = display('slurm status',display_id=True)
while True:
    # print human readable time
    time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    out = json.loads(subprocess.getoutput(f'sacct --json -S {START_TIME}'))
    jobs = {j['job_id']: j for j in out['jobs']}
    jobs = {k: j for k, j in jobs.items() if j['name'].endswith('-' + UUID)}

    pending = running = failed = successful = 0
    for _, j in jobs.items():
        state = j['state']['current'][0]
        if state == 'PENDING':
            pending += 1
        elif state == 'RUNNING':
            running += 1
        elif state == 'FAILED':
            failed += 1
        elif state == 'COMPLETED':
            if j['exit_code']['status'][0] == 'SUCCESS':
                successful += 1
            else:
                failed += 1
        else:
            raise ValueError(f'Unknown state: {j["state"]}')
    dh.update(printed_str(f'''{time_str}
Pending: {pending}, Running: {running}, Failed: {failed}, Successful: {successful}
'''))
    if running + pending == 0:
        break
    time.sleep(2)
EOF

