#!/bin/bash

set -e

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: \$0 <dataset_name> <output_file_path>"
    exit 1
fi

# Assign command-line arguments to variables
dataset=$1
output_file=$2
log_file="$output_file/$dataset-MAPLE[few-shot].log"

if [ -f "$log_file" ]; then
    last_line=$(tail -n 1 "$log_file")
    if [ "$last_line" = "All done!" ]; then
        echo "该数据集已总结完毕."
        exit 0
    else
        > "$log_file"
    fi
else
    touch "$log_file"
fi

# Experiment: MAPLE
echo "---------------------"
echo "$dataset dataset - MAPLE[few-shot]" 
echo "---------------------"

# Redirecting the output to the log file
exec > $log_file

# 16shot
echo "---------------------"
echo "now is 16shot traing."
bash scripts/maple/main.sh $dataset 16 > /dev/null
wait
echo "now the 16shot traing is done."
echo "---------------------"

# 8shot
echo "---------------------"
echo "now is 8shot traing."
bash scripts/maple/main.sh $dataset 8 > /dev/null
wait
echo "now the 8shot traing is done."
echo "---------------------"

# 4shot
echo "---------------------"
echo "now is 4shot traing."
bash scripts/maple/main.sh $dataset 4 > /dev/null
wait
echo "now the 4shot traing is done."
echo "---------------------"

# 2shot
echo "---------------------"
echo "now is 2shot traing."
bash scripts/maple/main.sh $dataset 2 > /dev/null
wait
echo "now the 2shot traing is done."
echo "---------------------"

# 1shot
echo "---------------------"
echo "now is 1shot traing."
bash scripts/maple/main.sh $dataset 1 > /dev/null
wait
echo "now the 1shot traing is done."
echo "---------------------"

# Wait for all experiments to finish before parsing results

# Parse and print results
echo "---------------------"
echo "Parsing results..."
echo "---------------------"

# 16shot
echo "---------------------"
echo "now is 16shot parsing."
python parse_test_res.py OUTPUT/MaPLe/${dataset}/vit_b16_c2_ep5_batch4_2ctx_16shots --end_signal "Finish training" | grep "* accuracy"  
wait
echo "now the 16shot parsing is done."
echo "---------------------"

# | grep -v "(" | grep "* accuracy" >> $log_file

# 8shot
echo "---------------------"
echo "now is 8shot parsing."
python parse_test_res.py OUTPUT/MaPLe/${dataset}/vit_b16_c2_ep5_batch4_2ctx_8shots --end_signal "Finish training" | grep "* accuracy"
wait
echo "now the 8shot parsing is done."
echo "---------------------"

# 4shot
echo "---------------------"
echo "now is 4shot parsing."
python parse_test_res.py OUTPUT/MaPLe/${dataset}/vit_b16_c2_ep5_batch4_2ctx_4shots --end_signal "Finish training" | grep "* accuracy" 
wait
echo "now the 4shot parsing is done."
echo "---------------------"

# 2shot
echo "---------------------"
echo "now is 2shot parsing."
python parse_test_res.py OUTPUT/MaPLe/${dataset}/vit_b16_c2_ep5_batch4_2ctx_2shots --end_signal "Finish training" | grep "* accuracy" 
wait
echo "now the 2shot parsing is done."
echo "---------------------"

# 1shot
echo "---------------------"
echo "now is 1shot parsing."
python parse_test_res.py OUTPUT/MaPLe/${dataset}/vit_b16_c2_ep5_batch4_2ctx_1shots --end_signal "Finish training" | grep "* accuracy"
wait
echo "now the 1shot parsing is done."
echo "---------------------"

echo "All done!"