#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: \$0 <dataset_name> <output_file_path>"
    exit 1
fi

# Assign command-line arguments to variables
dataset=$1
output_file=$2
log_file="$output_file/$dataset-COOP[few-shot]-CLIP[zero-shot].log"

# Experiment: COOP
echo "---------------------"
echo "$dataset dataset - COOP[few-shot] & CLIP[zero-shot]" 
echo "---------------------"

# Redirecting the output to the log file
exec > $log_file

# 16shot
echo "---------------------"
echo "now is 16shot traing."
bash scripts/coop/main.sh $dataset rn50 end 16 16 False > /dev/null
wait
echo "---------------------"

# 8shot
echo "---------------------"
echo "now is 8shot traing."
bash scripts/coop/main.sh $dataset rn50 end 16 8 False > /dev/null
wait
echo "---------------------"

# 4shot
echo "---------------------"
echo "now is 4shot traing."
bash scripts/coop/main.sh $dataset rn50_ep100 end 16 4 False > /dev/null
wait
echo "---------------------"

# 2shot
echo "---------------------"
echo "now is 2shot traing."
bash scripts/coop/main.sh $dataset rn50_ep100 end 16 2 False > /dev/null
wait
echo "---------------------"

# 1shot
echo "---------------------"
echo "now is 1shot traing."
bash scripts/coop/main.sh $dataset rn50_ep50 end 16 1 False > /dev/null
wait
echo "---------------------"

# Wait for all experiments to finish before parsing results

# Parse and print results
echo "---------------------"
echo "Parsing results..."
echo "---------------------"

# 16shot
echo "---------------------"
echo "now is 16shot parsing."
python parse_test_res.py /home/yzq/yzq_code/multimodal-prompt-learning-main/output/$dataset/CoOp/rn50_16shots/nctx16_cscFalse_ctpend --end_signal "Finish training" | grep "* accuracy"  
wait
echo "---------------------"

# | grep -v "(" | grep "* accuracy" >> $log_file

# 8shot
echo "---------------------"
echo "now is 8shot parsing."
python parse_test_res.py /home/yzq/yzq_code/multimodal-prompt-learning-main/output/$dataset/CoOp/rn50_8shots/nctx16_cscFalse_ctpend --end_signal "Finish training" | grep "* accuracy"  
wait
echo "---------------------"

# 4shot
echo "---------------------"
echo "now is 4shot parsing."
python parse_test_res.py /home/yzq/yzq_code/multimodal-prompt-learning-main/output/$dataset/CoOp/rn50_ep100_4shots/nctx16_cscFalse_ctpend --end_signal "Finish training" | grep "* accuracy"  
wait
echo "---------------------"

# 2shot
echo "---------------------"
echo "now is 2shot parsing."
python parse_test_res.py /home/yzq/yzq_code/multimodal-prompt-learning-main/output/$dataset/CoOp/rn50_ep100_2shots/nctx16_cscFalse_ctpend --end_signal "Finish training" | grep "* accuracy"  
wait
echo "---------------------"

# 1shot
echo "---------------------"
echo "now is 1shot parsing."
python parse_test_res.py /home/yzq/yzq_code/multimodal-prompt-learning-main/output/$dataset/CoOp/rn50_ep50_1shots/nctx16_cscFalse_ctpend --end_signal "Finish training" | grep "* accuracy" 
wait
echo "---------------------"

# CLIP_ZERO_SHOT
echo "---------------------"
echo "now is CLIP_ZERO_SHOT."
bash scripts/zsclip/zeroshot.sh $dataset rn50 | grep "* accuracy" 
echo "---------------------"